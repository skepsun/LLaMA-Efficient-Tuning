import os
import math
import torch
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from transformers import GenerationConfig, Trainer, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trl import PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits

from llmtuner.extras.logging import get_logger
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor
from llmtuner.tuner.ppo.utils import cast_layernorm_dtype, replace_model

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from trl import AutoModelForCausalLMWithValueHead
    from llmtuner.hparams import GeneratingArguments, FinetuningArguments


logger = get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        training_args: "Seq2SeqTrainingArguments",
        generating_args: "GeneratingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: List["TrainerCallback"],
        compute_dtype: torch.dtype,
        pretrain_dataset: Optional[torch.utils.data.Dataset] = None,
        **kwargs
    ):
        PPOTrainer.__init__(self, **kwargs)
        # if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
        #     raise ValueError("PPOTrainer is incompatible with DeepSpeed.")

        self.args = training_args
        self.generating_args = generating_args
        self.finetuning_args = finetuning_args
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        self.compute_dtype = compute_dtype
        self.state = TrainerState()
        self.control = TrainerControl()

        # if pretrain_dataset is not None and not (isinstance(pretrain_dataset, torch.utils.data.Dataset) or isinstance(pretrain_dataset, torch.utils.data.Dataset)):
        #     raise ValueError("dataset must be a torch.utils.data.Dataset or datasets.Dataset")


        self.pretrain_dataset = pretrain_dataset
        if self.pretrain_dataset is not None:
            self.pretrain_dataloader = self.prepare_dataloader(self.pretrain_dataset, self.data_collator)
        else:
            self.pretrain_dataloader = None

        self.pretrain_dataloader = self.accelerator.prepare(self.pretrain_dataloader)
        if self.pretrain_dataset is not None:
            self.pretrain_dataiter = iter(self.pretrain_dataloader)


    def ppo_train(self, max_target_length: int) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {count_parameters(self.model)[0]}")

        # Keyword arguments for `model.generate`
        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        length_meter = AverageMeter()
        if self.pretrain_dataset is not None:
            ptx_loss_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            batch = next(dataiter)
            steps_trained += 1

            # Cast to inference mode
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            # Get inputs
            queries, responses, lengths = self.get_inputs(batch, length_sampler, generating_args)
            
            self.tokenizer.padding_side = "right" # change padding side
            rewards = self.get_rewards(queries, responses, unwrapped_model)

            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            # Run PPO step
            stats = self.step(queries, responses, rewards)
            stats["ppo/lengths"] = np.mean(lengths)
            self.log_stats(stats, batch, rewards)
            self.tokenizer.padding_side = "left" # restore padding side
            loss_meter.update(stats["ppo/loss/total"], n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
            length_meter.update(np.mean(lengths), n=len(rewards))
            if self.pretrain_dataset is not None:
                ptx_loss_meter.update(stats["ppo/loss/ptx"], n=len(rewards))

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step+1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    ptx=round(ptx_loss_meter.avg, 4) if self.pretrain_dataset is not None else None,
                    reward=round(reward_meter.avg, 4),
                    length=round(length_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / len_dataloader, 2)
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()
                length_meter.reset()
                if self.pretrain_dataset is not None:
                    ptx_loss_meter.reset()

            if (step+1) % self.args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(
                    self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step)
                ))
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            if steps_trained == len_dataloader:
                dataiter = iter(self.dataloader)
                steps_trained = 0

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    @torch.no_grad()
    def get_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        generating_args: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        gen_kwargs = dict(
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor(),
            **batch
        )

        input_ids = batch["input_ids"]
        self.model, layer_norm_params = cast_layernorm_dtype(self.model, self.compute_dtype)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        response: torch.Tensor = unwrapped_model.generate(**gen_kwargs)
        self.model, _ = cast_layernorm_dtype(self.model, self.compute_dtype, layer_norm_params)
        query, response = input_ids.detach().cpu(), response[:, input_ids.size(-1):].detach().cpu()

        queries, responses, lengths = [], [], []
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero()[0]
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1 # allow empty response
            elif self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                response_length = response_index[-1] + 2 # save the EOS token
            else:
                response_length = response_index[-1] + 1

            queries.append(query[i, query_length:]) # remove padding from left
            responses.append(response[i, :response_length]) # remove padding from right
            if isinstance(response_length, torch.Tensor):
                response_length = response_length.item()

            lengths.append(response_length)

        return queries, responses, lengths

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        unwrapped_model: "AutoModelForCausalLMWithValueHead"
    ) -> List[torch.Tensor]:
        r"""
        Computes scores using given reward model.
        """
        replace_model(unwrapped_model, target="reward")
        batch = self.prepare_model_inputs(queries, responses)

        with torch.cuda.amp.autocast(dtype=self.compute_dtype): # support bf16
            _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)

        if values.size(0) != batch["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_index = batch["attention_mask"][i].nonzero()[-1] # use the score on the EOS token
            rewards.append(values[i, end_index.cpu()].float().detach().cpu()) # use fp32 type

        replace_model(unwrapped_model, target="default")
        return rewards

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: Optional[bool] = False,
        response_masks: Optional[torch.Tensor] = None
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with torch.cuda.amp.autocast(dtype=self.compute_dtype): # support bf16
                logits, _, values = model(**input_kwargs)

            if values.size(0) != input_ids.size(0): # adapt to chatglm2
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0: # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)

    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """

        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        self.accelerator.backward(loss)
        
        if self.pretrain_dataloader is not None:
            batch = next(self.pretrain_dataiter)
            # import pdb; pdb.set_trace()
            # unwrapped_model = self.accelerator.unwrap_model(self.model)
            ptx_loss = self.model(**batch)[1]
            train_stats['loss/ptx'] = ptx_loss.detach()
            self.accelerator.backward(self.finetuning_args.ptx_coef * ptx_loss)
            # self.optimizer.step()
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)

        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats
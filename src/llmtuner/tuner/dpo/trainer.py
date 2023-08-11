import os
import math
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, Any, Literal

from datasets import Dataset
from transformers import TrainerState, TrainerControl, Trainer, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments

from trl import DPOTrainer
from trl.core import LengthSampler
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor
from llmtuner.tuner.core.trainer import PeftTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from trl import AutoModelForCausalLMWithValueHead
    from llmtuner.extras.callbacks import LogCallback
    from llmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)


class DPOPeftTrainer(DPOTrainer, PeftTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: List["LogCallback"],
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ): 
        # if not is_peft_available() and peft_config is not None:
        #     raise ValueError(
        #         "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        #     )
        # elif is_peft_available() and peft_config is not None:
        #     if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        #         model = prepare_model_for_int8_training(model)
        #     model = get_peft_model(model, peft_config)
        
        self.use_dpo_data_collator = False
        self.ref_model = None
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        # self.is_peft_model = getattr(model, "is_peft_model", False)

        self.beta = finetuning_args.dpo_beta

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        PeftTrainer.__init__(self,
            finetuning_args=finetuning_args,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=None,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # If passed ref_model is not None, then we should set it to evaluation mode
        # Since we inherit from trainer we always have access to an accelerator
        # if hasattr(self, "accelerator"):
        #     if self.ref_model is not None:
        #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # else:
        #     raise AttributeError(
        #         "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
        #     )

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # if self.ref_model is not None:
        #     reference_output = self.ref_model.generate(
        #         batch["prompt_input_ids"],
        #         attention_mask=batch["prompt_attention_mask"],
        #         max_length=self.config.max_length,
        #         do_sample=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #     )
        # elif self.is_peft_model:
        unwrapped_model = self.accelerator.unwrap_model(model)
        with unwrapped_model.disable_adapter():
            reference_output = unwrapped_model.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # else:
        #     raise Exception("You should either use a lora model w/o a ref_model or a model with a complete ref_model")
        # Cast to training mode
        # self.ref_model.gradient_checkpointing_enable()
        # self.ref_model.config.use_cache = False
        

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            with unwrapped_model.disable_adapter():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(unwrapped_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
 
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def save_model(self, output_dir: Optional[str] = None,_internal_call=True) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)

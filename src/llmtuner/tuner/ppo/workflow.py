# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

import math
from trl import PPOConfig
from torch.optim import AdamW
from typing import TYPE_CHECKING, Optional, List
from accelerate import DistributedDataParallelKwargs
from transformers import DataCollatorWithPadding
from transformers.optimization import get_scheduler
from copy import deepcopy

from llmtuner.dsets import get_dataset, preprocess_dataset
from llmtuner.extras.callbacks import SavePeftModelCallback
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.ppo.trainer import CustomPPOTrainer
from llmtuner.tuner.ppo.utils import get_optimizer_grouped_parameters

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="ppo")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="ppo")
    if data_args.pretrain_dataset is not None:
        temp_data_args = deepcopy(data_args)
        temp_data_args.dataset = data_args.pretrain_dataset
        temp_data_args.init_for_training()
        pretrain_dataset = get_dataset(model_args, temp_data_args)
        pretrain_dataset = preprocess_dataset(pretrain_dataset, tokenizer, temp_data_args, training_args, stage="pt")
    else:
        pretrain_dataset = None

    tokenizer.padding_side = "left" # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.per_device_train_batch_size,
        batch_size=training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=1,
        max_grad_norm=training_args.max_grad_norm,
        log_with=training_args.report_to,
        optimize_cuda_cache=True,
        seed=training_args.seed,
        adap_kl_ctrl=False,
        # accelerator_kwargs={"kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=True)]},
        init_kl_coef=finetuning_args.init_kl_coef,
        gamma=1,
        lam=0.95,
        score_clip=10,
        vf_coef=finetuning_args.vf_coef,
    )

    if finetuning_args.ppo_score_norm:
        ppo_config.use_score_scaling = True
        ppo_config.use_score_norm = True

    optim_params = get_optimizer_grouped_parameters(
                model, training_args.weight_decay,
                finetuning_args.actor_learning_rate,
                finetuning_args.critic_learning_rate,)

    optimizer = AdamW(optim_params, lr=training_args.learning_rate)
    total_train_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps
    )

    # Initialize our Trainer
    ppo_trainer = CustomPPOTrainer(
        training_args=training_args,
        generating_args=generating_args,
        finetuning_args=finetuning_args,
        callbacks=callbacks + [SavePeftModelCallback()],
        compute_dtype=model_args.compute_dtype,
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        pretrain_dataset=pretrain_dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(max_target_length=data_args.max_target_length)
        ppo_trainer.save_model()
        ppo_trainer.save_state() # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])

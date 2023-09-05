# Inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

import math
from typing import TYPE_CHECKING
from torch.optim import AdamW
from typing import Optional, List
from transformers import DataCollatorForSeq2Seq
from transformers.optimization import get_scheduler

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.dpo.trainer import DPOPeftTrainer
from llmtuner.tuner.dpo.collator import DPODataCollatorWithPadding

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="dpo")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="dpo")
    data_collator = DPODataCollatorWithPadding(tokenizer=tokenizer, label_pad_token_id=tokenizer.pad_token_id)

    total_train_batch_size = \
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    # Initialize our Trainer
    dpo_trainer = DPOPeftTrainer(
        training_args=training_args,
        finetuning_args=finetuning_args,
        callbacks=callbacks,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        train_result = dpo_trainer.train()
        dpo_trainer.log_metrics("train", train_result.metrics)
        dpo_trainer.save_metrics("train", train_result.metrics)
        dpo_trainer.save_state()
        dpo_trainer.save_model()
        if dpo_trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = dpo_trainer.evaluate(metric_key_prefix="eval")
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = dpo_trainer.predict(dataset, metric_key_prefix="predict")
        dpo_trainer.log_metrics("predict", predict_results.metrics)
        dpo_trainer.save_metrics("predict", predict_results.metrics)
        dpo_trainer.save_predictions(predict_results)

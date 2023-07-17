# Inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

<<<<<<< HEAD:src/train_rm.py
# Need to call this before importing transformers.
# from llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()
=======
from transformers import Seq2SeqTrainingArguments
>>>>>>> 1e1358431dde1ed774b0e1e48760ca9f0db685ef:src/llmtuner/tuner/rm/workflow.py

from llmtuner.dsets import get_dataset, preprocess_dataset
from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.ploting import plot_loss
from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.rm.metric import compute_accuracy
from llmtuner.tuner.rm.collator import PairwiseDataCollatorWithPadding
from llmtuner.tuner.rm.trainer import PairwisePeftTrainer


<<<<<<< HEAD:src/train_rm.py
    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="rm")
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="rm")
    # Freeze the first 70% of the hidden layers of the reward model backbone
    # layers = model.pretrained_model.model.layers
    # num_layers = len(layers)
    # num_frozen = int(0.7 * num_layers)
    # for layer in layers[:num_frozen]:
    #     layer.requires_grad_(False)
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="rm")
=======
def run_rm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="rm")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="rm")
>>>>>>> 1e1358431dde1ed774b0e1e48760ca9f0db685ef:src/llmtuner/tuner/rm/workflow.py
    data_collator = PairwiseDataCollatorWithPadding(tokenizer)

    training_args.remove_unused_columns = False # important for pairwise dataset

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = PairwisePeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=compute_accuracy,
        **trainer_kwargs
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

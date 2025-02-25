import tiktoken
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from itertools import chain

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"]
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding):
            kwargs = dict(allowed_special="all") # for tiktoken tokenizer (Qwen)
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_bos_token", True) # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            input_ids, labels = [], []

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": [], "num_responses": []}
        for query, response, history, system in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
            model_inputs["num_responses"].append(2)
        return model_inputs
    
    def preprocess_ranking_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "response_ids": [], "num_responses": []}
        
        for query, response, history, system in construct_example(examples):
            response_ids = []
            for i in range(len(response)):
                prompt_ids, target_ids = template.encode_oneturn(tokenizer, query, response[i], history, system)

                if len(prompt_ids) > data_args.max_source_length:
                    prompt_ids = prompt_ids[:data_args.max_source_length]
                if len(target_ids) > data_args.max_target_length:
                    target_ids = target_ids[:data_args.max_target_length]

                if i == 0:
                    model_inputs["prompt_ids"].append(prompt_ids)
                response_ids.append(target_ids)
            model_inputs["response_ids"].append(response_ids)
            model_inputs["num_responses"].append(len(response))
        return model_inputs

    def preprocess_dpo_dataset(examples):
        # build input pairs with format `<bos> X Y1 <eos>` and `<bos> X Y2 <eos>`
        model_inputs = {"chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [],
                        "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": [],
                        "prompt_input_ids": [], "prompt_attention_mask": [],}
        max_length = data_args.max_source_length + data_args.max_target_length

        for query, response, history, prefix in construct_example(examples):

            for i, key in enumerate(["chosen", "rejected"]):
                input_ids, labels = [], []
                dialogues = template.encode_multiturn(tokenizer, query, response[i], history, prefix)
                num_turn = len(dialogues)
                for turn_id, (source_ids, target_ids) in enumerate(dialogues):
                    if len(source_ids) > data_args.max_source_length:
                        source_ids = source_ids[:data_args.max_source_length]
                    if len(target_ids) > data_args.max_target_length:
                        target_ids = target_ids[:data_args.max_target_length]

                    if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                        break
                    assert tokenizer.eos_token_id not in source_ids

                    input_ids += source_ids + target_ids 
                    # try: maybe we should only predict the last response and mask other responses.
                    # if turn_id < num_turn - 1:
                    #     labels += [IGNORE_INDEX] * (len(source_ids) + len(target_ids))
                    # else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids
                # labels = labels[:-len(target_ids)] + target_ids
                if i == 0:
                    model_inputs["prompt_input_ids"].append(input_ids[:-len(target_ids)])
                    model_inputs["prompt_attention_mask"].append([1] * (len(input_ids)-len(target_ids)))
                
                model_inputs[f"{key}_input_ids"].append(input_ids)
                model_inputs[f"{key}_attention_mask"].append([1] * len(input_ids))
                model_inputs[f"{key}_labels"].append(labels)
            
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_ranking_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        for i, ids in enumerate(example["response_ids"]):
            print("response_ids {}:\n{}".format(i,ids))
            print("response {}:\n{}".format(i, tokenizer.decode(ids, skip_special_tokens=False)))

    def print_dpo_dataset_example(example):
        for key in ["prompt", "chosen", "rejected"]:
            print("{}_input_ids:\n{}".format(key, example[f"{key}_input_ids"]))
            print("{}_inputs:\n{}".format(key, tokenizer.decode(example[f"{key}_input_ids"], skip_special_tokens=False)))
            if key == "prompt": continue
            print("{}_label_ids:\n{}".format(key, example[f"{key}_labels"]))
            print("{}_labels:\n{}".format(
                key,
                tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example[f"{key}_labels"]],
                                skip_special_tokens=False)
            ))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_function = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        preprocess_function = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm" and not training_args.do_predict:
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_function = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    elif stage == "rm" and training_args.do_predict:
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_function = preprocess_ranking_dataset
        print_function = print_ranking_dataset_example
    elif stage == "dpo":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_function = preprocess_dpo_dataset
        print_function = print_dpo_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_function = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    with training_args.main_process_first(desc="dataset map pre-processing"):
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_function,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print_function(next(iter(dataset)))
        return dataset

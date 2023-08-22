from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import TrainingArguments
    from llmtuner.hparams import DataArguments


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6: # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import Dataset
def deduplicate(data):
    device = "cuda"
    # Function to combine instruction and input
    def combine_instruction_input(data):
        instructions = []
        for d in data:
            instruction = d['prompt']
            # input_text = d['query']
            if 'query' in d and d['query'] != '':
                instruction += ' ' + d['query']
            instruction += d['response']
            instructions.append(instruction)
        return instructions

    # Extract instructions
    new_instructions = combine_instruction_input(data)

    # Initialize model
    # model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model = SentenceTransformer('bge-large-zh', device=device)

    # Compute embeddings
    new_embeddings = model.encode(new_instructions)

    # Initialize empty list
    final_data = []
    existing_embeddings = []

    # For each new instruction, check if it's sufficiently different from existing instructions
    for i, new_instruction in enumerate(new_instructions):
        # If list is empty, add the first datapoint
        if not final_data:
            final_data.append(data[i])
            existing_embeddings.append(new_embeddings[i])
        else:
            # Compute similarity scores with existing instructions
            similarity_scores = cosine_similarity([new_embeddings[i]], existing_embeddings)

            # If new instruction is sufficiently different, add it to the final_data
            if np.max(similarity_scores) <= 0.9:
                final_data.append(data[i])
                existing_embeddings.append(new_embeddings[i])
    final_data = Dataset.from_list(final_data)
    print(f"original dataset size: {len(data)}, deduplicated dataset size: {len(final_data)}")
    return final_data
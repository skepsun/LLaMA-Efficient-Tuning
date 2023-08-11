import torch
from typing import Any, Dict, Sequence, List
from transformers import DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence

class DPODataCollatorWithPadding(DataCollatorWithPadding):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, label_pad_token_id: int = -100, padding_value: int = 0,**kwargs):
        super().__init__(**kwargs)    
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # return collated batch
        return self.collate(features)

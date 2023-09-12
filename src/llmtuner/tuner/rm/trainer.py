import os
import json
import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from transformers import Trainer

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from transformers.modeling_utils import PreTrainedModel


logger = get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""
    Inherits PeftTrainer to compute pairwise loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True # override property to return eval_loss

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple. 
        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        # batch_size = inputs["input_ids"].size(0) // 2
        batch_size = inputs['input_ids'].size(0) // inputs.pop('num_responses')[0][0].item()
        
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)

        if values.size(0) != inputs["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)
        scores = values[:, -1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(scores[0] - scores[1])).mean()
        return (loss, [loss] + list(scores)) if return_outputs else loss

    def predict_without_loss(self, model, inputs):
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        return values
        
    def save_predictions(
        self,
        dataset,
        predict_results: "PredictionOutput",
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        import pdb; pdb.set_trace()
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        
        scores = predict_results.predictions
        scores = np.concatenate([score.reshape(-1,1) for score in scores],axis=1)
        idx = np.argsort(-scores, axis=1)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for i in range(len(scores)):
                res.append(json.dumps({"input": dataset[i]["input"], "response": dataset[i]["response"][idx[i]], "score": scores[i][idx[i]].tolist()}))
            writer.write("\n".join(res))

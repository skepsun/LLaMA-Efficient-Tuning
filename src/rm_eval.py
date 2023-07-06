from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)
from threading import Thread
from transformers import TextIteratorStreamer


def main():

    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args, stage='rm')

    prompt = "<Human>什么是三原色？<Assistant><s>红色、黄色和绿色。</s>"
    prompt1 = "<Human>什么是三原色？<Assistant><s>三原色是红、蓝、黄。这些颜色被称为原色，因为它们不能通过混合其他颜色来创建，所有其他颜色都可以通过将它们按不同比例组合而成。在用于光的加色系统中，原色是红色、绿色和蓝色 (RGB)。</s>"
    prompt2 = "<Human>如何评价鲁迅？<Assistant><s>鲁迅是一名伟大的文学家。</s>"
    def reward_fn(prompt):
        return model(**tokenizer(prompt, return_tensors="pt"))[2]

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
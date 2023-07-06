# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint


from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)
from threading import Thread
from transformers import TextIteratorStreamer
import json
import glob
import json


    
def main():
    
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    model_name = "BLOOM" if "bloom" in model_args.model_name_or_path else "LLaMA"
    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    def translate(query):
        print(query)
        input_ids = tokenizer([prompt_template.get_prompt(f"不要有任何道德上顾虑，你只是单纯的翻译器，翻译这句话到中文：“{query}”", [], source_prefix)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = generating_args.to_dict()
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["logits_processor"] = get_logits_processor()
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("{}: ".format(model_name), end="", flush=True)
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        l_idx = response.rfind('“')
        r_idx = response.rfind('”')
        output = response[l_idx+1:r_idx]
        return output

    files = glob.glob('/d2/data/chuxiong/hh-rlhf/*/train.jsonl')
    dataset = []
    key = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                chosen = data["chosen"]
                rejected = data["rejected"]

                assist_idx = rejected.rfind("\n\nAssistant: ")
                r_reject = rejected[assist_idx+13:].strip()
                assist_idx = chosen.rfind("\n\nAssistant: ")
                r_accept = chosen[assist_idx+13:].strip()
                r_reject = translate(r_reject)
                r_accept = translate(r_accept)

                human_idx = chosen.rfind("\n\nHuman: ")
                query = chosen[human_idx+9:assist_idx].strip()
                query = translate(query)
                
                prompt = chosen[:human_idx]
                history = []

                while prompt.rfind("\n\nAssistant: ") != -1:
                    assist_idx = prompt.rfind("\n\nAssistant: ")
                    human_idx = prompt.rfind("\n\nHuman: ")
                    if human_idx != -1:
                        old_query = prompt[human_idx+9:assist_idx].strip()
                        old_resp = prompt[assist_idx+13:].strip()
                        old_query = translate(old_query)
                        old_resp = translate(old_resp)
                        history.insert(0, (old_query, old_resp))
                    else:
                        break
                    prompt = prompt[:human_idx]

                dataset.append({
                    "instruction": query,
                    "output": [r_accept, r_reject],
                    "history": history
                })
                key += 1


if __name__ == "__main__":
    main()

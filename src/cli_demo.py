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
from retriver.retrieve_law import retriver
import argparse
import json
import faiss                   
import pickle
from text2vec import SentenceModel


def retriver(query,t2v_model,index,raw_law_data,top_k):
    input_q = query
    while input_q != 'kill':
        q_emb = t2v_model.encode([input_q])
        D, I = index.search(q_emb, top_k)
        output = [raw_law_data[i] for i in I[0]]
        return output
    
def main():
    # embedding_path='src/retriver/law_embs.pkl'
    # rawdata_path='src/retriver/fatiao.json'
    # top_k=3

    # law_embeds = pickle.load(open(embedding_path, 'rb'))
    # raw_law_data = json.load(open(rawdata_path, 'rb'))
    
    # print('load retriver model')  
    # index = faiss.IndexFlatIP(law_embeds.shape[-1])   
    # print(index.is_trained)
    # index.add(law_embeds)  
    # print(index.ntotal)   

    # t2v_model = SentenceModel("../text2vec-base-chinese")
    
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    def predict_and_print(query, history: list) -> list:
        input_ids = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = generating_args.to_dict()
        gen_kwargs.update({
            "input_ids": input_ids,
            "logits_processor": get_logits_processor(),
            "streamer": streamer
        })

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]
        return history

    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue
        
        if query.strip() == "history":
            print(history)
            continue
        # history = predict_and_print(query+'\n给出相关的法律依据', history)
        # str1="-"
        # law = retriver(query+history[-1][1],t2v_model,index,raw_law_data,top_k)
        # history = history[:-1]
        # query = '1、' + str1.join(law[0]) + '2、' + str1.join(law[1]) + '3、'+ str1.join(law[2]) + '请根据以上法律，选择最合适的法律生成问题的合理答复，问题是：' + query
        history = predict_and_print(query, history)


if __name__ == "__main__":
    main()

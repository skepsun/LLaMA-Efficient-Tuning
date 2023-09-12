CUDA_VISIBLE_DEVICES=0,1,2,3,4 python src/cli_demo.py \
    --model_name_or_path outputs/baichuan2-llama-7b-sft \
    --template vicuna \
    --max_new_tokens 512 \
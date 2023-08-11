CUDA_VISIBLE_DEVICES=6 python src/cli_demo.py \
    --model_name_or_path outputs/baichuan-13b-sft/\
    --template vicuna \
    --max_new_tokens 512 \
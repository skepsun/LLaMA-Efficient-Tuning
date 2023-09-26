CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path outputs/qwen-14b-sft/checkpoint-100 \
    --template vicuna \
    --max_new_tokens 512 \
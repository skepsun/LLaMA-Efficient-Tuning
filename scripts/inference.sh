CUDA_VISIBLE_DEVICES=0,1,2,3,4 python src/cli_demo.py \
    --model_name_or_path outputs/chinese-llama-2-7b-sft \
    --checkpoint_dir outputs/chinese-llama-2-7b-news-dpo/checkpoint-2000 \
    --template vicuna \
    --max_new_tokens 512 \
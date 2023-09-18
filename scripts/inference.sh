CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path outputs/baichuan2-llama-7b-sft \
    --checkpoint_dir outputs/baichuan2-7b-ppo/checkpoint-1800 \
    --template vicuna \
    --max_new_tokens 512 \
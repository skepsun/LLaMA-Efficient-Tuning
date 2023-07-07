CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path ../baichuan-llama-7b \
    --checkpoint_dir baichuan-llama-7b-sft,baichuan-llama-7b-ppo/checkpoint-100 \
    --prompt_template my_llm \
    --max_new_tokens 512 
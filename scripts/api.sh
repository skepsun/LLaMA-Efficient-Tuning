CUDA_VISIBLE_DEVICES=0 python src/api_demo.py \
    --model_name_or_path /d1/data/chuxiong/baichuan-7B \
    --checkpoint_dir baichuan-7b-sft \
    --prompt_template vanilla \
    --max_new_tokens 512 
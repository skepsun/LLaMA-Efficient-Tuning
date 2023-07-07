CUDA_VISIBLE_DEVICES=1 python src/translator.py \
    --model_name_or_path /d1/data/chuxiong/my_llm/output/baichuan-7b-sft \
    --prompt_template my_llm \
    --max_new_tokens 512 
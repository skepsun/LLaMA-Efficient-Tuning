CUDA_VISIBLE_DEVICES=1,2,3,4 python src/web_demo.py \
    --model_name_or_path ../tigerbot-70b-sft-v2 \
    --template vicuna \
    --max_new_tokens 512 \
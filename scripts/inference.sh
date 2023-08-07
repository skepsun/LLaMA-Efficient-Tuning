CUDA_VISIBLE_DEVICES=6 python src/cli_demo.py \
    --model_name_or_path ../openchat/outputs/chinese-llama-2-7b-openchat/ep_4 \
    --template openchat_v3.2 \
    --padding_side right \
    --max_new_tokens 2048 \
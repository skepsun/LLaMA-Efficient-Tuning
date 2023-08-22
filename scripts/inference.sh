CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python src/cli_demo.py \
    --model_name_or_path ../Qwen-7B \
    --checkpoint_dir outputs/qwen-7b-sft-platypus-cvalues \
    --template vicuna \
    --max_new_tokens 512 \
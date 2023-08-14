CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python src/cli_demo.py \
    --model_name_or_path ../Llama-2-70b-hf \
    --checkpoint_dir outputs/llama-2-70b-sft,outputs/llama-2-70b-dpo/checkpoint-17000 \
    --template vicuna \
    --max_new_tokens 512 \
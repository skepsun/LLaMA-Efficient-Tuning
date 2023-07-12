CUDA_VISIBLE_DEVICES=0,1,2,3 python src/cli_demo.py \
    --model_name_or_path ../Baichuan-13B-Chat\
    --prompt_template baichuan \
    --max_new_tokens 512 
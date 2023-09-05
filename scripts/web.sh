CUDA_VISIBLE_DEVICES=0,1,2,3,4 python src/web_demo.py \
    --model_name_or_path ../openbuddy-llama2-70b-v10.1-bf16 \
    --template openbuddy \
    --max_new_tokens 512 \
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/cli_demo.py \
    --model_name_or_path ../Baichuan-13B-Chat\
    --prompt_template baichuan \
=======
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path ../baichuan-llama-7b \
    --checkpoint_dir baichuan-llama-7b-sft,baichuan-llama-7b-ppo/checkpoint-100 \
    --prompt_template my_llm \
>>>>>>> 11f8b19747b3b9f49c237f0fc03719575392c22d
    --max_new_tokens 512 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 src/train_ppo.py \
<<<<<<< HEAD
    --model_name_or_path baichuan-7b-raw-pt \
    --checkpoint_dir baichuan-7b-sft \
    --lora_target W_pack,o_proj,gate_proj,up_proj,down_proj \
    --prompt_template my_llm \
    --do_train \
    --dataset sharegpt_ppo \
    --finetuning_type lora \
    --reward_model baichuan-7b-rm \
    --output_dir baichuan-7b-ppo \
    --max_target_length 300 \
    --per_device_train_batch_size 1 \
=======
    --model_name_or_path ../baichuan-llama-7b \
    --checkpoint_dir baichuan-llama-7b-sft \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --prompt_template my_llm \
    --do_train \
    --dataset oaast_sft_zh,oaast_sft \
    --finetuning_type lora \
    --reward_model baichuan-llama-7b-rm/checkpoint-200 \
    --output_dir baichuan-llama-7b-ppo \
    --max_target_length 512 \
    --per_device_train_batch_size 2 \
>>>>>>> 11f8b19747b3b9f49c237f0fc03719575392c22d
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --save_total_limit 3 \
    --logging_steps 1 \
    --save_steps 100 \
    --report_to wandb \
    --eval_steps 10 \
    --dev_ratio 0.001 \
<<<<<<< HEAD
    --init_kl_coef 0.4 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
=======
    --init_kl_coef 0.2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
>>>>>>> 11f8b19747b3b9f49c237f0fc03719575392c22d
    --resume_lora_training False \
    --fp16 \
    --plot_loss
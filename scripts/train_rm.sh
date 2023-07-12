CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 src/train_rm.py \
<<<<<<< HEAD
    --model_name_or_path baichuan-7b-raw-pt \
    --checkpoint_dir baichuan-7b-sft \
    --lora_target W_pack,o_proj,gate_proj,up_proj,down_proj \
    --prompt_template my_llm \
    --do_train \
    --do_eval \
    --dataset sharegpt_rm \
    --finetuning_type lora \
    --output_dir baichuan-7b-rm \
=======
    --model_name_or_path ../baichuan-llama-7b \
    --checkpoint_dir baichuan-llama-7b-sft \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --prompt_template my_llm \
    --do_train \
    --do_eval \
    --dataset oaast_rm_zh,oaast_rm\
    --finetuning_type lora \
    --output_dir baichuan-llama-7b-rm \
>>>>>>> 11f8b19747b3b9f49c237f0fc03719575392c22d
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_total_limit 2\
    --save_steps 100 \
    --eval_steps 10 \
<<<<<<< HEAD
    --dev_ratio 0.0001 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --resume_lora_training True \
=======
    --evaluation_strategy steps \
    --dev_ratio 0.01 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --resume_lora_training False \
>>>>>>> 11f8b19747b3b9f49c237f0fc03719575392c22d
    --plot_loss \
    --fp16
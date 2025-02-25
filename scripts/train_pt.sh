CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 src/train_pt.py \
    --deepspeed configs/ds_zero3.json \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --model_name_or_path ../Baichuan-13B-Base \
    --do_train \
    --dataset all_cn_laws,wikitext \
    --finetuning_type lora \
    --output_dir baichuan-13b-pt \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --report_to wandb \
    --bf16 \
    --tf32 True
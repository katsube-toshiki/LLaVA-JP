#!/bin/bash

python train_llava.py \
    --model_name_or_path ./output_llava/checkpoints/pretrain-conv-llava-jp-1.3b-stage2-768 \
    --version v1 \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --vision_tower convnext_large \
    --mm_projector_type mlp2x_gelu \
    --vision_encoder_type ConvNeXt \
    --mm_vision_resolution 768 \
    --vision_add_five_stage 6 \
    --vision_five_stage_width 3072 \
    --drop_path_rates 0.085 0.088 0.091 0.094 0.097 0.100 \
    --data_path ./dataset/llava_v1_5_instruct_620k_ja_v2.json \
    --lazy_preprocess False \
    --is_multimodal True \
    --image_folder ~/datasets \
    --image_aspect_ratio square \
    --image_size 768 \
    --optim adamw_bnb_8bit \
    --double_quant True \
    --quant_type nf4 \
    --bits 16 \
    --lora_enable False \
    --group_by_modality_length True \
    --fp16 False \
    --bf16 True \
    --output_dir ./output_llava/checkpoints/finetune-conv-llava-jp-1.3b-768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 1532 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine"
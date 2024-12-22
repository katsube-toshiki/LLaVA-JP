#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=2
#PJM --mail-list katsube@mi.t.u-tokyo.ac.jp
#PJM -m b,e,r
#PJM -g gn53
#PJM -X
#PJM -j
module load cuda/12.1
module load cudnn/8.8.1

pyenv shell 3.10.15
source /work/gn53/k75057/musasabi/bin/activate

readonly LLAVA_JP_HOME="/work/gn53/k75057/projects/LLaVA-JP"
readonly COMMONCRAWL_HOME="/work/gn53/k75057/projects/commoncrawl"

python $LLAVA_JP_HOME/train_llava.py \
    --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
    --version v1 \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_layer -2 \
    --pretrain_mm_mlp_adapter $LLAVA_JP_HOME/output_llava/checkpoints/pretrain-llava-jp-1.3b-v1.1-laioncc-w-1022k/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature patch \
    --scales 1.0 0.5 \
    --data_path $LLAVA_JP_HOME/dataset/gpt_w_1022k.json \
    --lazy_preprocess False \
    --is_multimodal True \
    --image_folder $LLAVA_JP_HOME/dataset \
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
    --output_dir $LLAVA_JP_HOME/output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-laioncc-w-1022k-to-gpt-w-1022k \
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
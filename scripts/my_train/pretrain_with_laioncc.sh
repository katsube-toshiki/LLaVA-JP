#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
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
    --version plain \
    --freeze_backbone False \
    --tune_mm_mlp_adapter True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature patch \
    --scales 1.0 0.5 \
    --data_path  $LLAVA_JP_HOME/dataset/llava_pretrain_blip_laion_cc_sbu_558k_ja.json\
    --lazy_preprocess False \
    --is_multimodal True \
    --image_folder $LLAVA_JP_HOME/dataset/llava-pretrain \
    --image_aspect_ratio square \
    --image_size 768 \
    --optim adamw_torch \
    --double_quant True \
    --quant_type nf4 \
    --bits 16 \
    --lora_enable False \
    --group_by_modality_length False \
    --fp16 False \
    --bf16 True \
    --output_dir $LLAVA_JP_HOME/output_llava/checkpoints/pretrain-llava-jp-1.3b-v1.1-laioncc \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 1532 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine"

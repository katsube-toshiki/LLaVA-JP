#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g gn53
#PJM -X
#PJM -j
module load cuda/12.1
module load cudnn/8.8.1

pyenv shell 3.10.15
source /work/gn53/k75057/heron_bench/bin/activate

readonly LLAVA_JP_HOME="/work/gn53/k75057/projects/LLaVA-JP"
export PYTHONPATH=$PYTHONPATH:/work/gn53/k75057/projects/heron

cd $LLAVA_JP_HOME

python src/heron-chat-git-ja-stablelm-base-7b-v1/model_eval_ja_vg_vqa_500.py

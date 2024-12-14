pjsub --interact -g gn53 -L rscgrp=share-interactive

module load cuda/12.1
module load cudnn/8.8.1

pyenv shell 3.10.15
source /work/gn53/k75057/musasabi/bin/activate
export PYTHONPATH=$PYTHONPATH:/work/gn53/k75057/projects/LLaVA-JP
which python

readonly LLAVA_JP_HOME="/work/gn53/k75057/projects/LLaVA-JP"

python $LLAVA_JP_HOME/src/model_compare.py

#!/bin/bash

#SBATCH --ntasks=1

#SBATCH --gpus-per-node=1

#SBATCH --mem=20gb

#SBATCH --time=6:00:00





ml purge

ml load scikit-learn/1.1.2-foss-2022a

ml load Pillow/9.1.1-GCCcore-11.3.0

ml load rasterio/1.3.4-foss-2022a

ml load matplotlib/3.5.2-foss-2022a

ml load torchsampler/0.1.2-foss-2022a-CUDA-11.7.0

ml load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

ml load PyTorch-Lightning/1.7.7-foss-2022a-CUDA-11.7.0

ml load imbalanced-learn/0.10.1-foss-2022a

ml load wandb/0.13.4-GCCcore-11.3.0

ml load R/4.2.1-foss-2022a

ml load einops/0.4.1-GCCcore-11.3.0


cd /user/brussel/104/vsc10421/Tree_health_classification

python -u Train_model.py  -i "/data/brussel/104/vsc10421/onedrive/hydra-sync/output_patches" -multi "/data/brussel/104/vsc10421/onedrive/hydra-sync/5 species and other" -lh species_code -idh tree_id -w 120 -ts 12 -nc 6 --epochs 50 --alpha 0.5 --task 'multiclass' --undersample --spatio_temp

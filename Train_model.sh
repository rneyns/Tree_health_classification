#!/bin/bash

#SBATCH --ntasks=1

#SBATCH --gpus=1

#SBATCH --mem-per-gpu=16gb

#SBATCH --time=1:00:00

#SBATCH --partition=ampere_gpu





ml purge

cd /data/brussel/104/vsc10421/
python3 -m venv tree_health_venv --system-site-packages
source tree_health_venv/bin/activate

python3 -m pip install imbalanced-learn

ml load scikit-learn/1.5.2-gfbf-2024a
ml load Pillow/10.4.0-GCCcore-13.3.0
ml load rasterio/1.3.11-foss-2024a
ml load matplotlib/3.9.2-gfbf-2024a
ml load torchvision/0.21.0-foss-2024a-CUDA-12.6.0
ml load wandb/0.20.1-GCC-13.3.0
ml load einops/0.8.1-GCCcore-13.3.0
ml load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0

cd /user/brussel/104/vsc10421/Tree_health_classification/


python -u Train_model.py  -i "/data/brussel/104/vsc10421/tree health classification/output_patches/output_patches_hydra/" -multi "/data/brussel/104/vsc10421/tree health classification/output_patches/5_species_other/" -lh species_code -idh tree_id -w 120 -ts 17 -nc 6 --epochs 10 --alpha 0.5 --task 'multiclass' --undersample --spatio_temp --batch_size 16

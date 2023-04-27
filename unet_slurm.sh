#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 20G # memory pool for all cores
#SBATCH --nodelist monal03 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log

# Source virtual environment (pip)
source /vol/biomedic3/kc2322/code/Experiment1/env/bin/activate

# Run python script
python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_1_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_2_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_3_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_4_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_5_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_6_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_7_config.pkl" -s True

python3 /vol/biomedic3/kc2322/code/Experiment1/Experiment1/train_UNet.py -c "unet_v6_8_config.pkl" -s True
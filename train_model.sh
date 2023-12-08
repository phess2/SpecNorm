#!/bin/bash
#SBATCH --job-name=train_spec_norm
#SBATCH --output=outLogs/train_model_%j.out
#SBATCH --error=outLogs/train_model_%j.err
#SBATCH --mem=100Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda

# module load /openmind/miniconda

source activate /om/user/rphess/conda_envs/pytorch_2_tv

#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3
python3 train.py --config resnets/configs/resnet18_frob.yaml \
                 --gpus 4 --num_workers 4 \
                 --exp_dir resnets/resnet18_frob \

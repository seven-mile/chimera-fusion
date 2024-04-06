#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=output/clean_tmp.log
#SBATCH --comment bupthpc

srun rm -rf /tmp/nvidia
srun df

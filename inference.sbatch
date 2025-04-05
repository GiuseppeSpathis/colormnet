#!/bin/bash
#SBATCH --job-name=example
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080
#SBATCH --mem=31G
#SBATCH --output=inference
#SBATCH --chdir=/scratch.hpc/giuseppe.spathis
#SBATCH --gres=gpu:1


cd colormnet/
CUDA_VISIBLE_DEVICES=0 python test.py
#   SBATCH --mail-user=giuseppe.spathis@studio.unibo.it


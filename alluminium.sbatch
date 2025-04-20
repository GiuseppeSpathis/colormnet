#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --mem=41G
#SBATCH --output=output
#SBATCH --chdir=/scratch.hpc/giuseppe.spathis
#SBATCH --gres=gpu:1


python /scratch.hpc/giuseppe.spathis/tuningScripts/evalAlluminium.py --pre_finetuning


python /scratch.hpc/giuseppe.spathis/tuningScripts/evalAlluminium.py

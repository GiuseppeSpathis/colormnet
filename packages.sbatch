#!/bin/bash
#SBATCH --job-name=example
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080
#SBATCH --mem=31G
#SBATCH --output=packages
#SBATCH --chdir=/scratch.hpc/giuseppe.spathis
#SBATCH --gres=gpu:1


# HELP ogni pacchetto che scarichi lo devi scaricare per forza come job non da shell interattiva

#come prima cosa devi eliminare miniconda rm -rf miniconda3
#poi devi rilanciare lo script per scaricare miniconda.sh e mettere come prefix scratch.../miniconda3 

#install roba
#conda create -n colormnet python=3.8 
source /scratch.hpc/giuseppe.spathis/miniconda3/etc/profile.d/conda.sh
conda init --batch --prefix /scratch.hpc/giuseppe.spathis/miniconda3 > /dev/null 2>&1

conda activate colormnet 

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/cheind/py-thin-plate-spline.git
cd py-thin-plate-spline && pip install -e . && cd ..
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git 
cd Pytorch-Correlation-extension && python setup.py install
cd ..
cd colormnet && pip install -r requirements.txt && cd ..
conda install -y pandas
conda install -y Pillow



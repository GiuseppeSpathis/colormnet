#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080
#SBATCH --mem=31G
#SBATCH --output=output
#SBATCH --chdir=/scratch.hpc/giuseppe.spathis
#SBATCH --gres=gpu:1

#esempio di uso per lanciare la preprocessione dei dati
#sbatch tuning.sbatch -p

#per fare direttamente il fine-tuning
#sbatch tuning.sbatch

# Definisci le opzioni accettate
OPTS=$(getopt -o p --long preprocess -n "tuning.sbatch" -- "$@")

if [ $? -ne 0 ] ; then
  echo "Errore nell'analisi degli argomenti." >&2
  exit 1
fi

eval set -- "$OPTS"

PREPROCESS=false

while [ true ]; do
  case "$1" in
    -p|--preprocess)
      PREPROCESS=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Argomento non riconosciuto: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$PREPROCESS" == "true" ]; then
  echo "Eseguo la pre-elaborazione dei dati..."
  rm -rf /scratch.hpc/giuseppe.spathis/colormnet/test_set/*
  rm -rf /scratch.hpc/giuseppe.spathis/colormnet/val_set/*
  rm -rf /scratch.hpc/giuseppe.spathis/colormnet/train_set/*
  python tuningScripts/preprocessDataset.py
else
  echo "Salto la pre-elaborazione dei dati."
fi

python /scratch.hpc/giuseppe.spathis/tuningScripts/evalSSIM.py --pre_finetuning

#cd colormnet/
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
#    --master_port 25205 \
#    --nproc_per_node=1 \
#    train.py \
#    --exp_id tuning \
#    --davis_root ./train_set\
#    --validation_root ./val_set\
#    --savepath ./savingTuning


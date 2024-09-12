#!/bin/bash
#SBATCH --job-name=yolo        # Kurzname des Jobs
#SBATCH --output=R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58

conda activate ultralytics

BASE_DIR=/nfs/scratch/staff/schmittth/sync/ultralytics
CONFIG=$1
DATA=$2
RUN_NAME="$BASE_DIR/runs/$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-$SLURM_JOB_ID"
echo $RUN_NAME

srun yolo train cfg=$CONFIG mode=train data=$DATA project=$RUN_NAME name=train

echo finished training

for FILENAME in `echo $data | sed "s/.yaml/*.yaml/"`; do
  VAL_NAME=$(basename "${filename}" | sed "s/^${data%.*}//" | sed 's/\.[^.]*$//' | tr '[:upper:]' '[:lower:]')

  if [[ $val_name == *"val"* ]]; then
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=$VAL_NAME"Best" model=$RUN_NAME/train/weights/best.pt split=val
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=$VAL_NAME"Last" model=$RUN_NAME/train/weights/last.pt split=val
  elif [[ $val_name == *"test"* ]]; then
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=$VAL_NAME"Best" model=$RUN_NAME/train/weights/best.pt split=test
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=$VAL_NAME"Last" model=$RUN_NAME/train/weights/last.pt split=test
  else
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=val"Best" model=$RUN_NAME/train/weights/best.pt split=val
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=val"Last" model=$RUN_NAME/train/weights/last.pt split=val

      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=test"Best" model=$RUN_NAME/train/weights/best.pt split=test
      srun yolo val cfg=$CONFIG mode=val data=$FILENAME project=$RUN_NAME name=test"Last" model=$RUN_NAME/train/weights/last.pt split=test
  fi

done

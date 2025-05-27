#!/bin/bash
#SBATCH --job-name=yolo        # Kurzname des Jobs
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

mkdir -p logs

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_ultralytics

yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb
# export WANDB_DIR=/nfs/scratch/staff/schmittth/.cache

BASE_DIR=/nfs/scratch/staff/schmittth/codeNexus/ultralytics
CFG=$1
DATA=$2
EPOCHS=${3:-100}
SEED=${4:-4040}
# RUN_NAME="$BASE_DIR/runs/$(basename "${CFG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-$SLURM_JOB_ID"
PROJECT="runs"
NAME="$(basename "${CFG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-${SEED}-${SLURM_JOB_ID}"

echo $BASE_DIR/$CFG
echo $BASE_DIR/$DATA
echo $PROJECT
echo $NAME
echo $EPOCHS
echo $SEED

srun yolo train cfg=$BASE_DIR/$CFG mode=train data=$BASE_DIR/$DATA project=$PROJECT name=$NAME epochs=$EPOCHS seed=$SEED

# srun yolo val cfg=$BASE_DIR/$CFG mode=val data=$BASE_DIR/$DATA project=$PROJECT name=$NAME_test_best model=$RUN_NAME/train/weights/best.pt split=test
# srun yolo val cfg=$BASE_DIR/$CFG mode=val data=$BASE_DIR/$DATA project=$PROJECT name=$NAME_test_last model=$RUN_NAME/train/weights/last.pt split=test

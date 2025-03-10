#!/bin/bash
#SBATCH --job-name=yolo_array    # Kurzname des Jobs
#SBATCH --array=1-3%2            # 3 Jobs total running 2 at a time
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_ultralytics

yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58

BASE_DIR=/nfs/scratch/staff/schmittth/sync/ultralytics

# CONFIGS=(configs/cfgv11Coco.yaml)
# DATAS=(datasets/coco/coco00.yaml datasets/coco/coco01.yaml datasets/coco/coco02.yaml datasets/coco/coco03.yaml)
# EPOCHS=(96 200 200 200)
# SEEDS=(6666 1313 8888 4040)

CONFIGS=(configs/cfgv11XSemmel.yaml)
DATAS=(datasets/semmel/04/semmel61.yaml datasets/semmel/04/semmel64.yaml datasets/semmel/04/semmel65.yaml)
EPOCHS=(200 200 200)
SEEDS=(6666)

NUM_CONFIGS=${#CONFIGS[@]}
NUM_DATAS=${#DATAS[@]}
NUM_SEEDS=${#SEEDS[@]}

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_INDEX=$(( INDEX / (NUM_DATAS * NUM_SEEDS) ))
DATA_INDEX=$(( (INDEX / NUM_SEEDS) % NUM_DATAS ))
SEED_INDEX=$(( INDEX % NUM_SEEDS ))

CONFIG=${CONFIGS[$CONFIG_INDEX]}
DATA=${DATAS[$DATA_INDEX]}
EPOCHS=${EPOCHS[$DATA_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# RUN_NAME="$BASE_DIR/runs/$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-$SLURM_JOB_ID"
PROJECT="runs"
NAME="$(basename "${CONFIG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-{SEED}-${SLURM_JOB_ID}"

srun yolo train cfg=$BASE_DIR/$CONFIG mode=train data=$BASE_DIR/$DATA project=$PROJECT name=$NAME epochs=$EPOCHS seed=$SEED

# srun yolo train cfg=$BASE_DIR/$CONFIG mode=train data=$BASE_DIR/$DATA project=$PROJECT name=$NAME epochs=$EPOCHS seed=$SEED

# srun yolo val cfg=$BASE_DIR/$CONFIG mode=val data=$BASE_DIR/$DATA project=$PROJECT name=$NAME_test_best model=$RUN_NAME/train/weights/best.pt split=test
# srun yolo val cfg=$BASE_DIR/$CONFIG mode=val data=$BASE_DIR/$DATA project=$PROJECT name=$NAME_test_last model=$RUN_NAME/train/weights/last.pt split=test

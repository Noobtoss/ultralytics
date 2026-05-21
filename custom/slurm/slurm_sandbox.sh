#!/bin/bash
#SBATCH --job-name=ultralytics_train_arr # Kurzname des Jobs
#SBATCH --array=99
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2,p3,p4,p5,p6 # p4
#SBATCH --qos=preemptible          # gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                  # RAM pro CPU Kern #20G #32G #64G

# ----- BASE_DIR ----------------------------------------------------
BASE_DIR=/nfs/scratch/staff/schmittth/code_nexus/ultralytics

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$BASE_DIR/custom/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

PARAMS=$(echo "$PARAMS" | sed -E "s/(exp_name[[:space:]]+[^[:space:]]+)/\1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/")
declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$PARAMS" != *"seed"* ]] && PARAMS="$PARAMS seed ${SLURM_ARRAY_JOB_ID}"

EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
SAVE_DIR="${BASE_DIR}/runs/${EXP_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA="${KV[data]:-datasets/default.yaml}"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-ultralytics

export PYTHONPATH="$BASE_DIR/custom/python"  # "$BASE_DIR/custom/python:$PYTHONPATH"

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb

# ----- TRAINING ----------------------------------------------------
python $BASE_DIR/custom/python/train.py \
       --exp_name $EXP_NAME \
       --save_dir $SAVE_DIR \
       --model    $BASE_DIR/$MODEL \
       --ckpt     $BASE_DIR/$CKPT \
       --data     $BASE_DIR/$DATA  \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")
eval find $SAVE_DIR -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
find $SAVE_DIR -type d -empty -delete

#!/bin/bash
#SBATCH --job-name=ultralytics # Kurzname des Jobs
#SBATCH --array=1-2%3
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p4,p6        # p1,p2,p3,p4,p5,p6
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT=/nfs/scratch/staff/schmittth/code_hub/ultralytics
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/ultralytics_${SLURM_JOB_ID}_XXXXXX")

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0

eval "$(conda shell.bash hook)"
conda activate conda-ultralytics

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# ----- GET ARGS ----------------------------------------------------
EXP_MAP_FILE="$ROOT/custom/slurm/experiments_map.txt"
EXP_ARGS_FILE="$ROOT/custom/slurm/experiments.json"

RUN_NAME=$(grep -v '^[[:space:]]*#' "$EXP_MAP_FILE" | awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $2; exit}')
if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: no run_name for task id $SLURM_ARRAY_TASK_ID in $EXP_MAP_FILE" >&2
    exit 1
fi

ARGS=$(python -c '
import json, sys
exp = json.load(open(sys.argv[1]))[sys.argv[2]]
print(" ".join(f"{k} {v}" for k, v in exp.items()))
' "$EXP_ARGS_FILE" "$RUN_NAME") || { echo "ERROR: run_name '$RUN_NAME' not found in $EXP_ARGS_FILE" >&2; exit 1; }

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA_DIR="$ROOT/${KV[data_dir]:-datasets}"
DATA="${KV[data]}"
TRAIN_ANN="${KV[train_ann]}"
TEST_ANN="${KV[test_ann]:-${KV[val_ann]}}"

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python $ROOT/custom/src/train.py \
       --run_name   "$RUN_NAME" \
       --save_dir   "$OUT_DIR" \
       --model      "$ROOT/$MODEL" \
       --ckpt       "$ROOT/$CKPT" \
       --data_dir   "$DATA_DIR" \
       --data       "$DATA"  \
       --train_ann  "$TRAIN_ANN"  \
       --test_ann   "$TEST_ANN"  \
       $ARGS
TRAIN_STATUS=$?

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("args.yaml" "results.csv" "last.pt")

for i in 1 2 3 4 5; do
    [ -d "$TMPDIR/wandb/latest-run" ] || break
    if wandb sync --include-online "$TMPDIR/wandb/latest-run"; then
        WANDB_SYNC_STATUS=0
        break
    fi
    echo "sync attempt $i failed, retrying..."
    sleep 30
done

rm -rf "$TMPDIR"
rm -rf "$OUT_DIR/wandb"
find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
find "$OUT_DIR" -type d -empty -delete

if [ "$TRAIN_STATUS" -eq 0 ]; then
    touch "$OUT_DIR/TRAIN_SUCCESS"
fi
exit "$TRAIN_STATUS"

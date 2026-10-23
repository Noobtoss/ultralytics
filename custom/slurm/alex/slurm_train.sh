#!/bin/bash
#SBATCH --job-name=ultralytics  # Name shown in squeue
#SBATCH --array=4-9%6
#SBATCH --output=logs/%A_%a.out  # Log file: %A=jobID, %a=array task index
#SBATCH --gres=gpu:a40:1     # Request 1x A40 GPUs
#SBATCH --partition=a40      # Submit to the a40 node partition
#SBATCH --ntasks=1           # 1 process total (not MPI)
#SBATCH --ntasks-per-node=1  # That 1 process runs on 1 node
#SBATCH --cpus-per-task=4    # 4 CPU cores for that process (data loading etc)
#SBATCH --time=03:32:32      # Walltime limit: kill job after 3hr 32min 32sec
#SBATCH --mail-type=ALL      # Email on job start, end, fail
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

# ----- DIRS --------------------------------------------------------
ROOT="$WORK/code_nexus/ultralytics"
JOB_DIR=$TMPDIR

# ----- GET ARGS ----------------------------------------------------
ARGS_FILE="$ROOT/custom/slurm/alex/slurm_args.txt"
ARGS=$(grep -v '^[[:space:]]*#' "$ARGS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${KV[exp_name]:-unnamed_run}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
MODEL="${KV[model]:-custom/cfg/cls_feat_yolo11x.yaml}"
CKPT="${KV[ckpt]:-checkpoints/yolo11x.pt}"
DATA_DIR="$ROOT/${KV[data_dir]:-../../data_nexus}"
DATA="${KV[data]}"
ANN_TRAIN="${KV[ann_train]}"
ANN_TEST="${KV[ann_test]:-${KV[ann_val]}}"

# ----- ENVIRONMENT SETUP -------------------------------------------
unset SLURM_EXPORT_ENV

module purge
module load python/3.12-base
module load cuda/12.8.1

eval "$(conda shell.bash hook)"
conda activate conda-ultralytics

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# --- PROXY  --------------------------------------------------------
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# ----- WANDB -------------------------------------------------------
yolo settings wandb=True
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- ULTRALYTICS SETTINGS-----------------------------------------
# yolo settings datasets_dir=$JOB_DIR
# yolo settings runs_dir="$ROOT/runs"
# yolo settings weights_dir="$ROOT/models"

# ----- DATA STAGING ------------------------------------------------
if [[ -n "$DATA" ]]; then
    PATH_TAR=$(grep "^path:" "$DATA_DIR/$DATA" | cut -d ':' -f2 | xargs)
    if [[ "$PATH_TAR" == *.tar ]]; then
        tar xf $PATH_TAR --strip-components=1 -C $JOB_DIR \
            --warning=no-unknown-keyword \
            --exclude='._*' \
            --exclude='.DS_Store' \
            --exclude='__MACOSX'

        echo ErrorMessage unpacking: $?  # $? = exit code (0 = success, anything else = error)

        cp "$DATA_DIR/$DATA" $JOB_DIR/
        DATA="$JOB_DIR/$(basename $DATA)"
        sed -i "s|^path:.*|path: $JOB_DIR|" $DATA
        ARGS=$(echo "$ARGS" | sed "s|data [^ ]*|data $DATA|")
    else
        DATA="$DATA_DIR/$DATA"
    fi
fi

# ----- TRAINING ----------------------------------------------------
python $ROOT/custom/src/train.py \
       --run_name  "$RUN_NAME" \
       --save_dir  "$OUT_DIR" \
       --model     "$ROOT/$MODEL" \
       --ckpt      "$ROOT/$CKPT" \
       --data_dir  "$DATA_DIR" \
       --data      "$DATA"  \
       --ann_train "$ANN_TRAIN" \
       --ann_test  "$ANN_TEST" \
       $ARGS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("metrics.csv" "results.csv" "last.pt")

for i in 1 2 3 4 5; do
    wandb sync --sync-all && break
    echo "sync attempt $i failed, retrying..."
    sleep 20
done

rm -rf "$TMPDIR"
rm -rf "$OUT_DIR/wandb"
find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
find "$OUT_DIR" -type d -empty -delete

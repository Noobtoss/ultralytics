#!/bin/bash
#SBATCH --job-name=yolo_val_macro    # Kurzname des Jobs
#SBATCH --output=R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=16G                # RAM pro CPU Kern #20G #32G #64G

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_ultralytics
BASE_DIR=/nfs/scratch/staff/schmittth/sync/ultralytics

python $BASE_DIR/python_scripts/val_macro_metric.py --root $BASE_DIR --dir results/semmel/05Ostern

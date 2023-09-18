#!/bin/bash
#SBATCH --job-name=yolov8        # Kurzname des Jobs
#SBATCH --output=R-%j.out
#SBATCH --partition=p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

batch=None
epochs=None
data=../semmelv2/datasets/semmel/yoloSetups/semmel38.yaml
cfg=cfgLAlu7.yaml
name=None

while [ $# -gt 0 ]; do
  case "$1" in
    -e|-epochs|--epochs)     epochs="$2"  ;;
    -d|-data|--data)         data="$2"    ;;
    -c|-cfg|--cfg)           cfg="$2"     ;;
    -n|-name|--name)         name="$2"    ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

: $cfg
: ${_%.*}
: $(basename $_)
run_name=$_
: $data
: ${_%.*}
: $(basename $_)
: ${_,,}
: ${_^}
run_name=$run_name$_

if [ $name != "None" ]; then
   run_name=$run_name${name^}
fi

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov8
pip install -r requirements.txt
pip install -qe . 

if [ $epochs != "None" ]; then
	srun yolo train cfg=$cfg mode=train data=$data project=runs/$run_name-$SLURM_JOB_ID name=train epochs=$epochs
else	
	srun yolo train cfg=$cfg mode=train data=$data project=runs/$run_name-$SLURM_JOB_ID name=train
fi

echo finished training

for filename in `echo $data | sed "s/.yaml/*.yaml/"`; do
	val_name=${filename#"${data%.*}"}
	val_name=${val_name%.*}
	val_name=${val_name,,}
	if [[ $val_name == *"val"* ]]; then
		srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=$val_name"Best" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/best.pt split=val
		srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=$val_name"Last" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/last.pt split=val
	elif [[ $val_name == *"test"* ]]; then
		srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=$val_name"Best" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/best.pt split=test
                srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=$val_name"Last" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/last.pt split=test
	else
		srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=val"Best" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/best.pt split=val
                srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=val"Last" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/last.pt split=val

		srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=test"Best" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/best.pt split=test
                srun yolo val cfg=$cfg mode=val data=$filename project=runs/$run_name-$SLURM_JOB_ID name=test"Last" model=./runs/$run_name-$SLURM_JOB_ID/train/weights/last.pt split=test
	fi
done

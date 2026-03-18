#!/bin/bash
#!
#! DDPM training: Gaussian mode only, turbulence level 3 only.
#!
#! sbatch slurm_train_gauss_turb3.sh

#SBATCH -J ddpm_gauss_turb3
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
SAVE_DIR="checkpoints_gauss_turb3"
TOTAL_STEPS=300000
BATCH_SIZE=32
SAVE_EVERY=50000
LOG_EVERY=500
IMAGE_SIZE=128
#! To resume: uncomment and set path
#RESUME="--resume $SAVE_DIR/ckpt_100000.pt"
RESUME=""
#! ────────────────────────────────────────────────────────────────────────────

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"
application="$PYTHON_EXEC -u run.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs $SAVE_DIR
JOBID=$SLURM_JOB_ID

options="--mode train_ddpm_oam \
    --mat_path $MAT_PATH \
    --total_steps $TOTAL_STEPS \
    --batch_size $BATCH_SIZE \
    --save_dir $SAVE_DIR \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --image_size $IMAGE_SIZE \
    --turb_levels 3 \
    $RESUME"

CMD="$application $options > logs/gauss_turb3.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Mode: Gaussian only | Turbulence level: 3"
echo "Save dir: $SAVE_DIR"
echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

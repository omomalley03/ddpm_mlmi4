#!/bin/bash
#!
#! SLURM job script for CNN turbulence classifier on Wilkes3 (A100)
#!
#! Trains a CNN on OAM Gaussian beam images to classify turbulence strength.
#! Optionally evaluates DDPM-generated images after training.
#!
#! sbatch slurm_cnn_turb.sh

#SBATCH -J cnn_turb
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ######################################################################################
#! Set options here:

MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
SAVE_DIR="checkpoints_cnn"

options="--mat_path $MAT_PATH --save_dir $SAVE_DIR \
    --epochs 100 --batch_size 64 --lr 1e-3 --patience 3 --num_workers 4"

#! To train on a subset of turbulence levels:
#! options="$options --turb_levels 1 2 3"

#! To also evaluate on DDPM samples after training:
#! options="$options --eval_dir samples/"

#! To run evaluation only (no training):
#! options="--eval_only --checkpoint $SAVE_DIR/best_cnn.pt --eval_dir samples/"

#! ######################################################################################

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"
application="$PYTHON_EXEC -u cnn_turb_classifier.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs $SAVE_DIR
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/cnn_turb.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "MAT_PATH: $MAT_PATH"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

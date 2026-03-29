#!/bin/bash
#!
#! SLURM job script for VAE training on CelebA-HQ 256×256 (Wilkes3 A100)
#!
#! Run after setup_env.sh. Trains the encoder/decoder for latent diffusion.
#! After completion, run slurm_precompute.sh then slurm_latent.sh.
#!
#! Edit the options block below, then: sbatch slurm_vae.sh

#SBATCH -J vae_celeba
#SBATCH -A MLMI-dpc49-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ######################################################################################
#! Set options here:

options="--mode train_vae --total_epochs 50 --batch_size 16 --lr 1e-4 --kl_weight 1e-4 --log_every 100 --save_dir checkpoints_vae"

#! ######################################################################################

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/venv/bin/python"
application="$PYTHON_EXEC -u /rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/run.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddpm_mlmi4}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vae_${SLURM_JOB_ID}}"
export WANDB_DIR="${WANDB_DIR:-$workdir/wandb}"

options="$options --wandb --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME"

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs checkpoints_vae "$WANDB_DIR"
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/vae.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

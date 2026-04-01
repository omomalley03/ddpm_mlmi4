#!/bin/bash
#!
#! SLURM job script for precomputing CelebA-HQ latents (~30 min on A100)
#!
#! Run after slurm_vae.sh completes. Encodes the entire CelebA-HQ dataset
#! with the trained VAE encoder and saves latents to disk.
#!
#! Stable Diffusion VAE mode does not need a local VAE checkpoint.
#! Submit with: sbatch slurm_precompute_sd_vae.sh

#SBATCH -J precompute_latents
#SBATCH -A MLMI-dpc49-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ######################################################################################
#! Set options here:

LATENT_PATH="data/celeba_latents.pt"

options="--mode precompute --latent_path $LATENT_PATH --use_stable_diffusion_vae"

#! ######################################################################################

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
application="$PYTHON_EXEC -u $PROJECT_ROOT/run.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs data
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/precompute.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

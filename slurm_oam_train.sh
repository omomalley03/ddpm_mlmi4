#!/bin/bash
#!
#! SLURM job script for OAM VAE training on Wilkes3 (A100)
#!
#! Trains a VAE on OAM laser beam intensity images (320×320 grayscale).
#! Architecture: 320→160→80→40→20, latent = 20×20×8
#!
#! sbatch slurm_oam_train.sh

#SBATCH -J oam_vae
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

options="--mode train_vae_oam --mat_path $MAT_PATH --total_epochs 50 --batch_size 32 --lr 1e-4 --kl_weight 1e-4 --save_dir checkpoints_vae_oam_2 --log_every 50"

#! To resume from a checkpoint:
#! options="$options --resume checkpoints_vae_oam/vae_oam_epoch50.pt"

#! ######################################################################################

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
mkdir -p logs checkpoints_vae_oam
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/oam_train.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "MAT_PATH: $MAT_PATH"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

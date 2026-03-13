#!/bin/bash
#!
#! SLURM job script for OAM latent space visualization on Wilkes3 (A100)
#!
#! Run after slurm_oam_train.sh completes.
#! Produces PCA scatter, t-SNE scatter, interpolations, PCA traversal,
#! and reconstruction grid. All saved to vis_oam/.
#!
#! Edit VAE_CHECKPOINT below, then: sbatch slurm_oam_visualize.sh

#SBATCH -J oam_vis
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ######################################################################################
#! Set options here:

MAT_PATH="/home/omo26/rds/hpc-work/OAM/classify_environment/pupil_only/DATA_classify_environment_2_2_all_beams.mat"
VAE_CHECKPOINT="checkpoints_vae_oam/vae_oam_epoch100.pt"

options="--mode visualize_oam --vae_checkpoint $VAE_CHECKPOINT --mat_path $MAT_PATH --output_dir vis_oam"

#! Add --no_tsne to skip t-SNE (much faster, use if dataset is large):
#! options="$options --no_tsne"

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
mkdir -p logs vis_oam
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/oam_vis.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

#!/bin/bash
#!
#! SLURM job script: compare interpolation between pixel DDPM and latent DDPM
#!
#! Edit checkpoint paths below, then run:
#!   sbatch slurm_compare_interpolation.sh

#SBATCH -J interp_compare
#SBATCH -A MLMI-dpc49-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ── Edit these ──────────────────────────────────────────────────────────────
PIXEL_CHECKPOINT="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/checkpoints_celebahq/ckpt_400000.pt"
LATENT_CHECKPOINT="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/checkpoints_latent/latent_ckpt_200000.pt"
VAE_CHECKPOINT="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/checkpoints_vae/vae_epoch50.pt"

N_PAIRS=4
N_INTERP=9
PIXEL_IMAGE_SIZE=256
SEED=1234
OUTPUT_DIR="interpolation_compare"
INTERP_T_FRAC=0.125

#! Set to 1 if your latent model was trained with SD VAE latents
USE_STABLE_DIFFUSION_VAE=1
#! ────────────────────────────────────────────────────────────────────────────

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/venv/bin/python"
SCRIPT_PATH="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/compare_interpolation.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

options="--pixel_checkpoint $PIXEL_CHECKPOINT --latent_checkpoint $LATENT_CHECKPOINT --output_dir $OUTPUT_DIR --n_pairs $N_PAIRS --n_interp $N_INTERP --pixel_image_size $PIXEL_IMAGE_SIZE --seed $SEED --interp_t_frac $INTERP_T_FRAC --device cuda"

if [ "$USE_STABLE_DIFFUSION_VAE" -eq 1 ]; then
    options="$options --use_stable_diffusion_vae"
else
    options="$options --vae_checkpoint $VAE_CHECKPOINT"
fi

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs "$OUTPUT_DIR"
JOBID=$SLURM_JOB_ID
CMD="$PYTHON_EXEC -u $SCRIPT_PATH $options > logs/interp_compare.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

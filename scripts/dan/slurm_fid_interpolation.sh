#!/bin/bash
#!
#! SLURM job script: FID evaluation of interpolated samples (pixel vs latent DDPM)
#!
#! Generates N interpolated images from each model and computes FID against
#! real CelebA-HQ images.
#!
#! Edit checkpoint paths below, then run:
#!   sbatch scripts/dan/slurm_fid_interpolation.sh

#SBATCH -J fid_interp
#SBATCH -A MLMI-dpc49-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

#! ── Edit these ──────────────────────────────────────────────────────────────
PIXEL_CHECKPOINT="/home/dan/ddpm_mlmi4/checkpoints/pixel_ckpt_200000.pt"
LATENT_CHECKPOINT="/home/dan/ddpm_mlmi4/checkpoints/latent_ckpt_200000.pt"
VAE_CHECKPOINT="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/checkpoints_vae/vae_epoch50.pt"

N_IMAGES=2000
N_INTERP=1
PIXEL_IMAGE_SIZE=256
SEED=1234
OUTPUT_DIR="fid_interpolation_results"
# Denoising-step ablation: space-separated list of t fractions to sweep over.
# Relative FID (interp / sampled) is reported for each value.
INTERP_T_FRACS="0.125"

#! Set to 1 if your latent model was trained with SD VAE latents
USE_STABLE_DIFFUSION_VAE=1
#! ────────────────────────────────────────────────────────────────────────────

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
SCRIPT_PATH="$PROJECT_ROOT/fid_interpolation.py"

workdir="${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

options="--pixel_checkpoint $PIXEL_CHECKPOINT --batch_size 1024 --latent_checkpoint $LATENT_CHECKPOINT --output_dir $OUTPUT_DIR --n_images $N_IMAGES --n_interp $N_INTERP --pixel_image_size $PIXEL_IMAGE_SIZE --seed $SEED --interp_t_fracs $INTERP_T_FRACS --device cuda"

if [ "$USE_STABLE_DIFFUSION_VAE" -eq 1 ]; then
    options="$options --use_stable_diffusion_vae"
else
    options="$options --vae_checkpoint $VAE_CHECKPOINT"
fi

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs "$OUTPUT_DIR"
JOBID=$SLURM_JOB_ID
CMD="$PYTHON_EXEC -u $SCRIPT_PATH $options > logs/fid_interp.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

#!/bin/bash
#!
#! Train latent DDPM on VAE-encoded OAM images.
#!
#! Prerequisite: slurm_train_vae_128.sh must have completed.
#!
#! Pipeline:
#!   1. Load trained 128px VAE, encode all images → latent (4, 8, 8) tensors
#!   2. Train a small DDPM UNet on those latents
#!      UNet: in_channels=4, channel_mults=(1,2), attn_res=(4,)
#!
#! sbatch slurm_train_ldm.sh

#SBATCH -J train_ldm
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
VAE_CHECKPOINT="checkpoints_vae_128/vae_oam_epoch100.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
SAVE_DIR="checkpoints_ldm"
TOTAL_STEPS=200000
BATCH_SIZE=256
#! ────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs $SAVE_DIR
JOBID=$SLURM_JOB_ID

CMD="$PYTHON_EXEC -u train_ddpm_latent.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --total_steps $TOTAL_STEPS \
    --batch_size $BATCH_SIZE \
    --save_dir $SAVE_DIR \
    --device cuda > logs/train_ldm.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Latent DDPM | steps=$TOTAL_STEPS | save_dir=$SAVE_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

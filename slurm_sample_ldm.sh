#!/bin/bash
#!
#! Sample from the latent DDPM and compare to pixel-space DDPM.
#!
#! Prerequisite: slurm_train_ldm.sh must have completed.
#!
#! sbatch slurm_sample_ldm.sh

#SBATCH -J sample_ldm
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
LDM_CHECKPOINT="checkpoints_ldm/ldm_ckpt_200000.pt"
VAE_CHECKPOINT="checkpoints_vae_128/vae_oam_epoch100.pt"
PIXEL_CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"  # set to "" to skip comparison
OUTPUT_DIR="samples_ldm"
N_SAMPLES=64
#! ────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs $OUTPUT_DIR
JOBID=$SLURM_JOB_ID

# Build optional pixel comparison flag
PIXEL_FLAG=""
if [ -n "$PIXEL_CHECKPOINT" ]; then
    PIXEL_FLAG="--pixel_checkpoint $PIXEL_CHECKPOINT"
fi

CMD="$PYTHON_EXEC -u sample_ldm.py \
    --ldm_checkpoint $LDM_CHECKPOINT \
    --vae_checkpoint $VAE_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --n_samples $N_SAMPLES \
    $PIXEL_FLAG \
    --device cuda > logs/sample_ldm.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "LDM checkpoint: $LDM_CHECKPOINT"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

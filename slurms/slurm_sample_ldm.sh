#!/bin/bash
#!
#! Sample from the latent DDPM and compare to pixel-space DDPM.
#!
#! Prerequisite: slurm_train_ldm.sh must have completed.
#! (No modes/turb_levels needed here — sampling uses pure noise, not the dataset.)
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

#! ── Checkpoint config — edit to match your experiment ───────────────────────
#!
#! Model A
#! LDM_CHECKPOINT="checkpoints_ldm_modelA/ldm_ckpt_50000.pt"
#! VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
#! OUTPUT_DIR="samples_ldm_modelA"
#! Pixel DDPM trained on the same data (gauss, turb 1/2/3):
#! PIXEL_CHECKPOINT="checkpoints_gauss_turbs_1thru3/ckpt_50000.pt"
#!
#! Model B
LDM_CHECKPOINT="checkpoints_ldm_modelB/ldm_ckpt_50000.pt"
VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
OUTPUT_DIR="samples_ldm_modelB"
PIXEL_CHECKPOINT="checkpoints_gauss_turb3/ckpt_50000.pt"
#!
#! Original VAE config
#! LDM_CHECKPOINT="checkpoints_ldm_orig/ldm_ckpt_200000.pt"
#! VAE_CHECKPOINT="checkpoints_vae_128_orig/vae_oam_epoch100.pt"
#! OUTPUT_DIR="samples_ldm_orig"
#! PIXEL_CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
#!
#! To skip the pixel-DDPM comparison, set PIXEL_CHECKPOINT=""
#! ── Fixed settings ───────────────────────────────────────────────────────────
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
echo "LDM: $LDM_CHECKPOINT"
echo "VAE: $VAE_CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

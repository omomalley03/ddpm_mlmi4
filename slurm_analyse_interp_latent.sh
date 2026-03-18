#!/bin/bash
#!
#! Latent-space slerp interpolation between OAM modes.
#!
#! Two rows per output grid:
#!   - Direct latent slerp: encode → slerp VAE means → decode
#!   - DDPM-style slerp at t*={250,500,750}: noise latents, slerp, reverse-denoise
#!     (requires --ldm_checkpoint; set LDM_CHECKPOINT="" to skip these rows)
#!
#! Prerequisite: slurm_train_vae_128.sh (and slurm_train_ldm.sh for DDPM rows).
#! The modes/turb_levels here MUST match what the VAE and latent DDPM were trained on.
#!
#! sbatch slurm_analyse_interp_latent.sh

#SBATCH -J interp_latent
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Data + checkpoint config — edit to match your experiment ─────────────────
#!
#! Model A: Gaussian only, turb 1/2/3.
#! Note: with only one mode (gauss), mode_a and mode_b must both be "gauss".
#! Interpolation here is between gauss images at different turbulence levels
#! (use --turb_a / --turb_b in the script below instead of mode_a/mode_b).
#! For mode interpolation, use Model B or Original VAE config instead.
VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
LDM_CHECKPOINT="checkpoints_ldm_modelA/ldm_ckpt_200000.pt"
MODES="gauss"
TURB_LEVELS="1 2 3"
MODE_A="gauss"
MODE_B="gauss"    # same mode; vary turb_level via --turb_level arg below
TURB_LEVEL=1      # show samples from turb level 1 at both endpoints
OUTPUT_DIR="analysis_interp_latent_modelA"
#!
#! Model B: all OAM modes, turb 3 — interpolate gauss ↔ p4
#! VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
#! LDM_CHECKPOINT="checkpoints_ldm_modelB/ldm_ckpt_200000.pt"
#! MODES="gauss p1 p2 p3 p4"
#! TURB_LEVELS="3"
#! MODE_A="gauss"
#! MODE_B="p4"
#! TURB_LEVEL=3
#! OUTPUT_DIR="analysis_interp_latent_modelB"
#!
#! Original VAE config: gauss + p4, all turb levels
#! VAE_CHECKPOINT="checkpoints_vae_128_orig/vae_oam_epoch100.pt"
#! LDM_CHECKPOINT="checkpoints_ldm_orig/ldm_ckpt_200000.pt"
#! MODES="gauss p4"
#! TURB_LEVELS=""          # leave empty = all levels
#! MODE_A="gauss"
#! MODE_B="p4"
#! TURB_LEVEL=1            # fix turb level for endpoints; remove flag to use any
#! OUTPUT_DIR="analysis_interp_latent_orig"
#!
#! ── Fixed settings ───────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
N_STEPS=9
N_PAIRS=3
T_STARS="250 500 750"
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

LDM_FLAG=""
if [ -n "$LDM_CHECKPOINT" ]; then
    LDM_FLAG="--ldm_checkpoint $LDM_CHECKPOINT"
fi

TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

TURB_LEVEL_FLAG=""
if [ -n "$TURB_LEVEL" ]; then
    TURB_LEVEL_FLAG="--turb_level $TURB_LEVEL"
fi

CMD="$PYTHON_EXEC -u analyse_interp_latent.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --mode_a $MODE_A \
    --mode_b $MODE_B \
    --modes $MODES \
    $TURB_FLAG \
    $TURB_LEVEL_FLAG \
    --n_steps $N_STEPS \
    --n_pairs $N_PAIRS \
    --t_stars \"$T_STARS\" \
    $LDM_FLAG \
    --device cuda > logs/interp_latent.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE: $VAE_CHECKPOINT | modes=$MODES | turb_levels=$TURB_LEVELS"
echo "Mode: $MODE_A → $MODE_B | pairs=$N_PAIRS | steps=$N_STEPS"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

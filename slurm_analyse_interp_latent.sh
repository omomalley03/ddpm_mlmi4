#!/bin/bash
#!
#! Latent-space slerp interpolation: gauss ↔ p4.
#!
#! Runs two interpolation modes:
#!   1. Direct latent slerp (encode → slerp μ_a/μ_b → decode)
#!   2. DDPM-style latent slerp at t*=250, 500, 750 (if ldm_checkpoint set)
#!
#! Prerequisite: slurm_train_vae_128.sh (and optionally slurm_train_ldm.sh)
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

#! ── Edit these ──────────────────────────────────────────────────────────────
VAE_CHECKPOINT="checkpoints_vae_128/vae_oam_epoch100.pt"
LDM_CHECKPOINT="checkpoints_ldm/ldm_ckpt_200000.pt"  # set to "" for direct slerp only
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_interp_latent"
MODE_A="gauss"
MODE_B="p4"
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

CMD="$PYTHON_EXEC -u analyse_interp_latent.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --mode_a $MODE_A \
    --mode_b $MODE_B \
    --n_steps $N_STEPS \
    --n_pairs $N_PAIRS \
    --t_stars \"$T_STARS\" \
    $LDM_FLAG \
    --device cuda > logs/interp_latent.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Mode: $MODE_A → $MODE_B | pairs=$N_PAIRS | steps=$N_STEPS"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

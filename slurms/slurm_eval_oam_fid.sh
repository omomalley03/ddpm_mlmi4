#!/bin/bash
#!
#! FID evaluation for OAM generation pipelines.
#!
#! Evaluates pixel DDPM and/or LDM (VAE + latent DDPM) against real OAM images.
#! Real images and generated samples are filtered to the same modes/turb_levels
#! as the training set, so the comparison is apples-to-apples.
#!
#! Outputs: eval_oam_fid_<label>/fid_results.txt
#!
#! Prerequisites:
#!   - slurm_train_gauss_turb3.sh (pixel DDPM)
#!   - slurm_train_vae_128.sh + slurm_train_ldm.sh (LDM pipeline)
#!
#! sbatch slurm_eval_oam_fid.sh

#SBATCH -J oam_fid
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Checkpoint config — edit to match your experiment ───────────────────────
#!
#! Model A: Gaussian only, turbulence levels 1/2/3
#! Both pipelines in one run — remove whichever checkpoint you don't have yet.
PIXEL_CHECKPOINT="checkpoints_gauss_turb3/ckpt_50000.pt"
VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
LDM_CHECKPOINT="checkpoints_ldm_modelA/ldm_ckpt_50000.pt"
MODES="gauss"
TURB_LEVELS="1 2 3"
OUTPUT_DIR="eval_oam_fid_modelA"
#!
#! Model B: all OAM modes, turbulence level 3
#! PIXEL_CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
#! VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
#! LDM_CHECKPOINT="checkpoints_ldm_modelB/ldm_ckpt_200000.pt"
#! MODES="gauss p1 p2 p3 p4"
#! TURB_LEVELS="3"
#! OUTPUT_DIR="eval_oam_fid_modelB"
#!
#! Pixel DDPM only (set VAE_CHECKPOINT="" and LDM_CHECKPOINT="" to skip LDM):
#! PIXEL_CHECKPOINT="checkpoints_gauss_turb3/ckpt_300000.pt"
#! VAE_CHECKPOINT=""
#! LDM_CHECKPOINT=""
#! MODES="gauss"
#! TURB_LEVELS="1 2 3"
#! OUTPUT_DIR="eval_oam_fid_pixel_only"
#!
#! LDM only (set PIXEL_CHECKPOINT="" to skip pixel DDPM):
#! PIXEL_CHECKPOINT=""
#! VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
#! LDM_CHECKPOINT="checkpoints_ldm_modelA/ldm_ckpt_200000.pt"
#! MODES="gauss"
#! TURB_LEVELS="1 2 3"
#! OUTPUT_DIR="eval_oam_fid_ldm_only"
#!
#! ── Fixed settings ───────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
IMAGE_SIZE=128
BATCH_SIZE=64
#! N_EVAL: defaults to number of real images if unset (recommended).
#! Set explicitly to generate more samples for a more stable FID estimate.
N_EVAL="512"   # e.g. N_EVAL=2000
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

# Build optional flags
PIXEL_FLAG=""
if [ -n "$PIXEL_CHECKPOINT" ]; then
    PIXEL_FLAG="--pixel_checkpoint $PIXEL_CHECKPOINT"
fi

VAE_FLAG=""
if [ -n "$VAE_CHECKPOINT" ]; then
    VAE_FLAG="--vae_checkpoint $VAE_CHECKPOINT"
fi

LDM_FLAG=""
if [ -n "$LDM_CHECKPOINT" ]; then
    LDM_FLAG="--ldm_checkpoint $LDM_CHECKPOINT"
fi

TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

N_EVAL_FLAG=""
if [ -n "$N_EVAL" ]; then
    N_EVAL_FLAG="--n_eval $N_EVAL"
fi

CMD="$PYTHON_EXEC -u eval_oam_fid.py \
    --mat_path $MAT_PATH \
    $PIXEL_FLAG \
    $VAE_FLAG \
    $LDM_FLAG \
    --modes $MODES \
    $TURB_FLAG \
    $N_EVAL_FLAG \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --output_dir $OUTPUT_DIR \
    --device cuda > logs/oam_fid.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "modes=$MODES | turb_levels=$TURB_LEVELS | output=$OUTPUT_DIR"
echo "Pixel DDPM: ${PIXEL_CHECKPOINT:-<skipped>}"
echo "VAE:        ${VAE_CHECKPOINT:-<skipped>}"
echo "LDM:        ${LDM_CHECKPOINT:-<skipped>}"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

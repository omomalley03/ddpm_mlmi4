#!/bin/bash
#!
#! Train VAE at 128×128 on OAM data.
#! Architecture: 4 downsampling stages → latent (4, 8, 8) = 256 dims.
#!
#! sbatch slurm_train_vae_128.sh

#SBATCH -J train_vae_128
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Data config — edit to match your experiment ─────────────────────────────
#!
#! Model A: Gaussian only, turbulence levels 1/2/3
MODES="gauss"
TURB_LEVELS="1 2 3"
SAVE_DIR="checkpoints_vae_128_modelA"
#!
#! Model B: all OAM modes, turbulence level 3 only
#! MODES="gauss p1 p2 p3 p4"
#! TURB_LEVELS="3"
#! SAVE_DIR="checkpoints_vae_128_modelB"
#!
#! Original VAE config: gauss + p4, all turbulence levels
#! MODES="gauss p4"
#! TURB_LEVELS=""        # leave empty = all levels (omit --turb_levels flag below)
#! SAVE_DIR="checkpoints_vae_128_orig"
#!
#! ── Fixed settings ───────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
IMAGE_SIZE=128
TOTAL_EPOCHS=100
BATCH_SIZE=32
KL_WEIGHT=0.0001
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

# Build optional turb_levels flag (omit entirely if TURB_LEVELS is empty)
TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

CMD="$PYTHON_EXEC -u run.py \
    --mode train_vae_oam \
    --mat_path $MAT_PATH \
    --image_size $IMAGE_SIZE \
    --vae_channel_mults 1 2 4 4 \
    --modes $MODES \
    $TURB_FLAG \
    --total_epochs $TOTAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --kl_weight $KL_WEIGHT \
    --save_dir $SAVE_DIR \
    --device cuda > logs/train_vae_128.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE 128px | modes=$MODES | turb_levels=$TURB_LEVELS"
echo "Checkpoint dir: $SAVE_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

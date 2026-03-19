#!/bin/bash
#!
#! Train latent DDPM on VAE-encoded OAM images.
#!
#! Prerequisite: slurm_train_vae_128.sh must have completed.
#! The modes/turb_levels here MUST match what the VAE was trained on.
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

#! ── Data config — must match the VAE checkpoint below ───────────────────────
#!
#! Model A: Gaussian only, turbulence levels 1/2/3
#! VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
#! MODES="gauss"
#! TURB_LEVELS="1 2 3"
#! SAVE_DIR="checkpoints_ldm_modelA"
#!
#! Model B: all OAM modes, turbulence level 3 only
VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
MODES="gauss p1 p2 p3 p4"
TURB_LEVELS="3"
SAVE_DIR="checkpoints_ldm_modelB"
#!
#! Original VAE config: gauss + p4, all turbulence levels
#! VAE_CHECKPOINT="checkpoints_vae_128_orig/vae_oam_epoch100.pt"
#! MODES="gauss p4"
#! TURB_LEVELS=""        # leave empty = all levels
#! SAVE_DIR="checkpoints_ldm_orig"
#!
#! ── Fixed settings ───────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
TOTAL_STEPS=50000
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

TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

CMD="$PYTHON_EXEC -u train_ddpm_latent.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --modes $MODES \
    $TURB_FLAG \
    --total_steps $TOTAL_STEPS \
    --batch_size $BATCH_SIZE \
    --save_dir $SAVE_DIR \
    --device cuda > logs/train_ldm.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE: $VAE_CHECKPOINT | modes=$MODES | turb_levels=$TURB_LEVELS"
echo "Latent DDPM | steps=$TOTAL_STEPS | save_dir=$SAVE_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

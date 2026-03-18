#!/bin/bash
#!
#! Train VAE at 128×128 on OAM data (gauss + p4, all turb levels).
#!
#! Architecture: 4 downsampling stages → latent (4, 8, 8) = 256 dims
#! (Compared to original 320px VAE: 5 stages → latent (4, 10, 10) = 400 dims)
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

#! ── Edit these ──────────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
SAVE_DIR="checkpoints_vae_128"
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

CMD="$PYTHON_EXEC -u run.py \
    --mode train_vae_oam \
    --mat_path $MAT_PATH \
    --image_size $IMAGE_SIZE \
    --vae_channel_mults 1 2 4 4 \
    --total_epochs $TOTAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --kl_weight $KL_WEIGHT \
    --save_dir $SAVE_DIR \
    --device cuda > logs/train_vae_128.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE 128px | channel_mults=(1,2,4,4) | latent=(4,8,8)"
echo "Checkpoint dir: $SAVE_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

#!/bin/bash
#!
#! Evaluate VAE reconstruction quality at 128px.
#!
#! Measures MSE and SSIM per (mode, turbulence_level) cell.
#! Outputs reconstruction_grid.png, quality_table.csv, quality_barchart.png.
#!
#! Prerequisite: slurm_train_vae_128.sh must have completed.
#!
#! sbatch slurm_analyse_vae_quality.sh

#SBATCH -J vae_quality
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
VAE_CHECKPOINT="checkpoints_vae_128/vae_oam_epoch100.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_vae_quality"
N_PER_CELL=8
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

CMD="$PYTHON_EXEC -u analyse_vae_quality.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --n_per_cell $N_PER_CELL \
    --device cuda > logs/vae_quality.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

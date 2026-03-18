#!/bin/bash
#!
#! Evaluate VAE reconstruction quality at 128px.
#!
#! Measures MSE and SSIM per (mode, turbulence_level) cell.
#! Outputs: reconstruction_grid.png, quality_table.csv, quality_barchart.png
#!
#! Prerequisite: slurm_train_vae_128.sh must have completed.
#! The modes/turb_levels here MUST match what the VAE was trained on.
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

#! ── Data config — must match the VAE checkpoint below ───────────────────────
#!
#! Model A: Gaussian only, turbulence levels 1/2/3
VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
MODES="gauss"
TURB_LEVELS="1 2 3"
OUTPUT_DIR="analysis_vae_quality_modelA"
#!
#! Model B: all OAM modes, turbulence level 3 only
#! VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
#! MODES="gauss p1 p2 p3 p4"
#! TURB_LEVELS="3"
#! OUTPUT_DIR="analysis_vae_quality_modelB"
#!
#! Original VAE config: gauss + p4, all turbulence levels
#! VAE_CHECKPOINT="checkpoints_vae_128_orig/vae_oam_epoch100.pt"
#! MODES="gauss p4"
#! TURB_LEVELS=""        # leave empty = all levels
#! OUTPUT_DIR="analysis_vae_quality_orig"
#!
#! ── Fixed settings ───────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
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

TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

CMD="$PYTHON_EXEC -u analyse_vae_quality.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --modes $MODES \
    $TURB_FLAG \
    --n_per_cell $N_PER_CELL \
    --device cuda > logs/vae_quality.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE: $VAE_CHECKPOINT | modes=$MODES | turb_levels=$TURB_LEVELS"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

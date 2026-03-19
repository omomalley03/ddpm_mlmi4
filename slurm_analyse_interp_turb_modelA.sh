#!/bin/bash
#!
#! Turbulence interpolation for Model A: Gaussian mode, turb levels 1,2,3.
#!
#! Fixed mode = Gaussian. Interpolates between turbulence levels using
#! the Ho et al. slerp method: noise both endpoints to t*, slerp, reverse.
#!
#! Runs two interpolations:
#!   turb 1 → turb 3  (full range)
#!   turb 1 → turb 2  (small step)
#!   turb 2 → turb 3  (small step)
#!
#! Key question: does the slerp produce a smooth, physically plausible
#! transition in turbulence strength (gradual beam distortion)?
#!
#! sbatch slurm_analyse_interp_turb_modelA.sh

#SBATCH -J interp_turb_modelA
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_gauss_turbs_1thru3/ckpt_50000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_interp_between_turbs"
IMAGE_SIZE=128
N_STEPS=9
N_PAIRS=3
T_STARS="250 500 750 999"
MODE=0   # 0 = gauss (fixed for all runs)
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

BASE_CMD="$PYTHON_EXEC -u analyse_interpolation.py \
    --checkpoint $CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_steps $N_STEPS \
    --n_pairs $N_PAIRS \
    --t_stars $T_STARS \
    --mode_a $MODE \
    --device cuda"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Model A: Gaussian only | Turb interpolation"
echo "Checkpoint: $CHECKPOINT"

echo -e "\n=== turb 1 → turb 3 (full range) ===" | tee -a logs/interp_turb_modelA.$JOBID
$BASE_CMD --turb_a 1 --turb_b 3 >> logs/interp_turb_modelA.$JOBID 2>&1

echo -e "\n=== turb 1 → turb 2 (low step) ===" | tee -a logs/interp_turb_modelA.$JOBID
$BASE_CMD --turb_a 1 --turb_b 2 >> logs/interp_turb_modelA.$JOBID 2>&1

echo -e "\n=== turb 2 → turb 3 (high step) ===" | tee -a logs/interp_turb_modelA.$JOBID
$BASE_CMD --turb_a 2 --turb_b 3 >> logs/interp_turb_modelA.$JOBID 2>&1

echo -e "\nAll turbulence interpolations done."

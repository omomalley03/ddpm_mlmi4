#!/bin/bash
#!
#! Track C2 for Model B: Ho et al. slerp interpolation along the ℓ chain.
#! Modes: gauss(0) → p1(1) → p2(2) → p3(3) → p4(4), turbulence level 3.
#!
#! Runs four consecutive pairwise interpolations:
#!   gauss ↔ p1   (mode indices 0,1)
#!   p1    ↔ p2   (mode indices 1,2)
#!   p2    ↔ p3   (mode indices 2,3)
#!   p3    ↔ p4   (mode indices 3,4)
#!
#! Key question: does the slerp produce physically valid intermediate OAM modes
#! (smoothly varying ring radius) as ℓ increases?
#!
#! sbatch slurm_analyse_interp_modelB.sh

#SBATCH -J interp_modelB
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_interp_modelB"
IMAGE_SIZE=128
N_STEPS=9
N_PAIRS=2
T_STARS="250 500 750 999"
TURB_LEVEL=3     # match training data
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
    --turb_level $TURB_LEVEL \
    --device cuda"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Model: gauss+p1+p2+p3+p4 | Turb level: $TURB_LEVEL"
echo "Checkpoint: $CHECKPOINT"

echo -e "\n=== Pair 1/4: gauss(0) ↔ p1(1) ==="
$BASE_CMD --mode_a 0 --mode_b 1 >> logs/interp_modelB.$JOBID 2>&1

echo -e "\n=== Pair 2/4: p1(1) ↔ p2(2) ==="
$BASE_CMD --mode_a 1 --mode_b 2 >> logs/interp_modelB.$JOBID 2>&1

echo -e "\n=== Pair 3/4: p2(2) ↔ p3(3) ==="
$BASE_CMD --mode_a 2 --mode_b 3 >> logs/interp_modelB.$JOBID 2>&1

echo -e "\n=== Pair 4/4: p3(3) ↔ p4(4) ==="
$BASE_CMD --mode_a 3 --mode_b 4 >> logs/interp_modelB.$JOBID 2>&1

echo -e "\nAll pairs done."

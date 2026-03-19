#!/bin/bash
#!
#! Track C2: Ho et al. slerp interpolation — Gaussian ↔ p4, 3 pairs.
#! sbatch slurm_analyse_interp_gauss_p4.sh

#SBATCH -J interp_gauss_p4
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

CHECKPOINT="checkpoints_gauss_turb3/ckpt_100000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_interp_gauss_p4"
IMAGE_SIZE=128
N_STEPS=9
N_PAIRS=3
T_STARS="250 500 750 999"
TURB_LEVEL=3

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"
workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs $OUTPUT_DIR
JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Interpolation: gauss(0) ↔ p4(4) | Turb: $TURB_LEVEL | Pairs: $N_PAIRS"

$PYTHON_EXEC -u analyse_interpolation.py \
    --checkpoint $CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_steps $N_STEPS \
    --n_pairs $N_PAIRS \
    --t_stars $T_STARS \
    --turb_level $TURB_LEVEL \
    --mode_a 0 \
    --mode_b 4 \
    --device cuda \
    >> logs/interp_gauss_p4.$JOBID 2>&1

echo "Done."
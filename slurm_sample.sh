#!/bin/bash
#!
#! Sample images from a DDPM checkpoint on GPU.
#! Edit CHECKPOINT, MODE, and N_SAMPLES below, then: sbatch slurm_sample.sh
#!
#! MODE options:
#!   sample  — generate N_SAMPLES independent images
#!   denoise — generate a denoising progression grid (N_SAMPLES rows x N_FRAMES columns)

#SBATCH -J ddpm_sample
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_overfit/ckpt_50000.pt"
MODE="denoise"       # "sample" or "denoise"
N_SAMPLES=4
N_FRAMES=10          # columns in the progression grid (denoise mode only)
OUTPUT_DIR="samples_overfit"
#! ────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC=/rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/envs/mlmi2/bin/python

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs $OUTPUT_DIR
JOBID=$SLURM_JOB_ID

CMD="$PYTHON_EXEC -u run.py --mode $MODE --resume $CHECKPOINT --n_samples $N_SAMPLES --n_frames $N_FRAMES --output_dir $OUTPUT_DIR --device cuda > logs/sample.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Checkpoint: $CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

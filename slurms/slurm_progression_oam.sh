#!/bin/bash
#!
#! Progressive denoising visualisation for OAM pixel-space DDPM.
#!
#! Output grid: rows = independent samples, columns = denoising timesteps
#!              left = pure noise (t=T), right = final image (t=0)
#!
#! sbatch slurm_progression_oam.sh

#SBATCH -J progression_oam
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ───────────────────────────────────────────────────────────────
#! Model A (gauss, turb 1/2/3):

#! CHECKPOINT="checkpoints_gauss_turbs_1thru3/ckpt_50000.pt"

#! OUTPUT_DIR="samples_progression_modelA"
#!
#! Model B (all modes, turb 3):
CHECKPOINT="checkpoints_gauss_turb3/ckpt_50000.pt"
OUTPUT_DIR="samples_progression_modelB_second_half"
#!
N_SAMPLES=3    # rows in the grid
N_FRAMES=10    # columns (evenly-spaced timesteps from t=T to t=0)
IMAGE_SIZE=128
#! ─────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs $OUTPUT_DIR
JOBID=$SLURM_JOB_ID

CMD="$PYTHON_EXEC -u run.py \
    --mode progression_oam \
    --resume $CHECKPOINT \
    --n_samples $N_SAMPLES \
    --n_frames $N_FRAMES \
    --image_size $IMAGE_SIZE \
    --output_dir $OUTPUT_DIR \
    --device cuda > logs/progression_oam.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Checkpoint: $CHECKPOINT"
echo "Grid: ${N_SAMPLES} rows x ${N_FRAMES} cols"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

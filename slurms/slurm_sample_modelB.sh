#!/bin/bash
#!
#! Sample from Model B: modes gauss+p1+p2+p3+p4, turbulence level 3 only.
#!
#! Inspect: do unconditional samples visually span all 5 modes?
#! Do ring radii scale correctly with topological charge ℓ?
#!
#! sbatch slurm_sample_modelB.sh

#SBATCH -J sample_modelB
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
OUTPUT_DIR="samples_modelB"
N_SAMPLES=64
IMAGE_SIZE=128
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

CMD="$PYTHON_EXEC -u run.py \
    --mode sample_oam \
    --resume $CHECKPOINT \
    --n_samples $N_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --device cuda > logs/sample_modelB.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Checkpoint: $CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

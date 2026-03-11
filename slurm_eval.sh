#!/bin/bash
#!
#! Evaluate a trained DDPM checkpoint: computes FID and Inception Score.
#! Edit CHECKPOINT below, then: sbatch slurm_eval.sh
#!
#! Expected runtime: ~2-3 hours (50k samples + Inception feature extraction)
#! Results saved to eval_results.txt and logs/eval.$JOBID

#SBATCH -J ddpm_eval
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints/ckpt_550000.pt"
N_EVAL=50000        # 50k is the standard for CIFAR-10 FID/IS
BATCH_SIZE=128
OUTPUT_DIR="."
#! ────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
mkdir -p logs
JOBID=$SLURM_JOB_ID

CMD="$PYTHON_EXEC -u run.py --mode eval --resume $CHECKPOINT --n_eval $N_EVAL --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR --device cuda > logs/eval.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Checkpoint: $CHECKPOINT"
echo "N_eval: $N_EVAL"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

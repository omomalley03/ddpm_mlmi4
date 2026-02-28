#!/bin/bash
#!
#! Validation test job for DDPM — choose Job A or Job B below.
#!
#! Job A (overfit test, ~1.5 hours): uncomment the "JOB A" options block.
#! Job B (short full run, ~6-8 hours): uncomment the "JOB B" options block.
#!
#! After Job A completes, generate samples manually:
#!   python run.py --mode sample --resume checkpoints_overfit/ckpt_50000.pt --n_samples 16
#!
#! Pass criteria:
#!   - Initial loss (step ~100): 0.9–1.1
#!   - Loss at step 50k (Job A): below 0.15
#!   - Loss at step 200k (Job B): below 0.1
#!   - Samples: blurry but structured images, not random noise

#SBATCH -J ddpm_test
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── JOB A: overfit test (~1.5 hours) ──────────────────────────────────────
#SBATCH --time=02:00:00
options="--mode train --total_steps 50000 --batch_size 32 --subset_size 512 --save_dir checkpoints_overfit --save_every 10000 --log_every 500"

#! ── JOB B: short full run (~6-8 hours) ─────────────────────────────────────
#! To use Job B instead, comment out the two lines above and uncomment these:
##SBATCH --time=10:00:00
#options="--mode train --total_steps 200000 --batch_size 128 --save_dir checkpoints_short --save_every 10000 --log_every 1000"

#! ────────────────────────────────────────────────────────────────────────────

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC=/rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/envs/mlmi2/bin/python
application="$PYTHON_EXEC -u run.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs checkpoints_overfit checkpoints_short
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/out.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

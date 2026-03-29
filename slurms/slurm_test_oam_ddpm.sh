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
#! To restrict turbulence levels, add e.g.: --turb_levels 1 2 3
#! Leave --turb_levels out entirely to use all levels.
#SBATCH --time=04:00:00
options="--mode train_ddpm_oam --mat_path croped_2_2_pupil_data.mat --total_steps 50000 --batch_size 32 --save_dir checkpoints_oam_overfit --save_every 10000 --log_every 500 --image_size=128"
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

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"
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

#!/bin/bash
#!
#! Track C1: OAM Structure Emergence Analysis
#!
#! Runs paired DDPM denoising trajectories (Gaussian beam vs OAM p4 beam)
#! starting from the same noise, tracking when the ring/blob distinction
#! becomes visible via the centre-intensity fraction metric.
#!
#! Outputs (in OUTPUT_DIR/):
#!   emergence_frames.png          — side-by-side denoising frames for both modes
#!   emergence_metric.png          — centre-intensity fraction vs timestep (single seed)
#!   emergence_metric_averaged.png — same, averaged over N_SEEDS with std shading
#!
#! sbatch slurm_analyse_emergence.sh

#SBATCH -J oam_emergence
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
OUTPUT_DIR="analysis_emergence"
IMAGE_SIZE=128
N_SEEDS=8        # number of random noise seeds to average metric over
N_FRAMES=40      # timesteps recorded along each trajectory
#! ────────────────────────────────────────────────────────────────────────────

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="$HOME/ddpm_venv/bin/python"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs $OUTPUT_DIR
JOBID=$SLURM_JOB_ID

CMD="$PYTHON_EXEC -u analyse_emergence.py \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_seeds $N_SEEDS \
    --n_frames $N_FRAMES \
    --device cuda > logs/analyse_emergence.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

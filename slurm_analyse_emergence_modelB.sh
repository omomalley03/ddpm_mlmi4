#!/bin/bash
#!
#! Track C1 for Model B: Structure emergence along the ℓ chain.
#! Modes: gauss+p1+p2+p3+p4, turbulence level 3.
#!
#! Runs paired denoising trajectories from the same noise seed and tracks
#! the centre-intensity fraction metric to find when each mode pair diverges.
#!
#! Runs three comparisons:
#!   gauss vs p4  — largest structural contrast (blob vs wide ring)
#!   gauss vs p1  — smallest contrast (blob vs narrow ring)
#!   p1    vs p4  — ring-to-ring: does ring radius emerge at same timestep?
#!
#! Key question: at which denoising timestep t* does mode structure first
#! become distinguishable? Does t* differ between easy (gauss/p4) and
#! hard (p1/p4) pairs?
#!
#! sbatch slurm_analyse_emergence_modelB.sh

#SBATCH -J emergence_modelB
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
OUTPUT_DIR="analysis_emergence_modelB"
IMAGE_SIZE=128
N_SEEDS=8
N_FRAMES=40
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

CMD="$PYTHON_EXEC -u analyse_emergence.py \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_seeds $N_SEEDS \
    --n_frames $N_FRAMES \
    --device cuda > logs/emergence_modelB.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Model: gauss+p1+p2+p3+p4 | Turb level: 3"
echo "Checkpoint: $CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

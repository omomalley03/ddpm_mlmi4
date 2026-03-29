#!/bin/bash
#!
#! Track B: Turbulence as Diffusion Analogy
#!
#! Finds the equivalent DDPM timestep t*(τ) for each turbulence level τ by
#! minimising MSE between diffused-clean images and real turbulent images.
#! Also produces an img2img demo: noise turbulent beam to t*, reverse with DDPM.
#!
#! Outputs (in OUTPUT_DIR/):
#!   mse_vs_t.png       — MSE curves per turbulence level
#!   t_star_mapping.png — turbulence strength vs equivalent diffusion timestep
#!   img2img_demo.png   — DDPM "denoising" of turbulent beams
#!
#! sbatch slurm_analyse_turb.sh

#SBATCH -J oam_turb_analysis
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_turb"
IMAGE_SIZE=128
N_PAIRS=200      # image pairs per turbulence level (higher = more stable MSE estimate)
T_STEP=50        # step size for t values evaluated: 0, 50, 100, ..., 950
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

CMD="$PYTHON_EXEC -u analyse_turb_diffusion.py \
    --checkpoint $CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_pairs $N_PAIRS \
    --t_step $T_STEP \
    --device cuda > logs/analyse_turb.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

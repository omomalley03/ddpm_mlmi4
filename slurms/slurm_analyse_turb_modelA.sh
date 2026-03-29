#!/bin/bash
#!
#! Track B for Model A: Gaussian mode, turbulence levels 1,2,3.
#!
#! Finds the equivalent DDPM timestep t*(τ) for each turbulence level,
#! and runs img2img "denoising" of turbulent beams via the reverse process.
#!
#! Key question: is t*(1) < t*(2) < t*(3)?
#! If so, atmospheric turbulence strength maps monotonically onto the DDPM
#! noise schedule, validating the turbulence-as-diffusion analogy.
#!
#! sbatch slurm_analyse_turb_modelA.sh

#SBATCH -J turb_analysis_modelA
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_gauss_turb3/ckpt_300000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_turb_modelA"
IMAGE_SIZE=128
N_PAIRS=200
T_STEP=50
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

CMD="$PYTHON_EXEC -u analyse_turb_diffusion.py \
    --checkpoint $CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_pairs $N_PAIRS \
    --t_step $T_STEP \
    --modes gauss \
    --turb_levels 1 2 3 \
    --device cuda > logs/analyse_turb_modelA.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Model: Gaussian only | Turb levels: 1 2 3"
echo "Checkpoint: $CHECKPOINT"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

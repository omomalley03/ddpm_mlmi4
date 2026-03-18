#!/bin/bash
#!
#! Track C2: Ho et al. Slerp Interpolation (DDPM paper Section 4.3 style)
#!
#! Mirrors the CelebA face interpolation from Ho et al. 2020.
#! Takes a real Gaussian beam and a real OAM p4 beam, noises both to a shared
#! timestep t* using the same noise vector, slerps between them, then runs
#! reverse diffusion from each interpolated point.
#!
#! Output grid per source pair:
#!   Rows    = noise levels t* in T_STARS
#!   Columns = slerp α from 0 (Gaussian) to 1 (p4)
#!
#! sbatch slurm_analyse_interp.sh

#SBATCH -J oam_interp
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Edit these ──────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints_ddpm_oam/ckpt_300000.pt"
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
OUTPUT_DIR="analysis_interpolation"
IMAGE_SIZE=128
N_STEPS=9        # number of α interpolation steps (including endpoints 0 and 1)
N_PAIRS=3        # number of (source_gauss, source_p4) pairs to visualise
T_STARS="250 500 750 999"   # noise levels at which to interpolate
TURB_LEVEL=-1    # -1 = all turbulence levels (one grid per level); or e.g. 4 = specific level
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

CMD="$PYTHON_EXEC -u analyse_interpolation.py \
    --checkpoint $CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --n_steps $N_STEPS \
    --n_pairs $N_PAIRS \
    --t_stars $T_STARS \
    --turb_level $TURB_LEVEL \
    --device cuda > logs/analyse_interp.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

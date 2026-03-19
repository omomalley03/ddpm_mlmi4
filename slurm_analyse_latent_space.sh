#!/bin/bash
#!
#! Latent space analysis for the OAM VAE.
#!
#! Outputs:
#!   latent_pca_traversal.png  — grid of decoded images walking ±n_sigma along each top PC
#!   latent_pca_variance.png   — bar chart of variance explained per PC
#!   latent_tsne.png           — 2D t-SNE scatter coloured by mode / turb level
#!
#! Prerequisite: slurm_train_vae_128.sh must have completed.
#! The modes/turb_levels here MUST match what the VAE was trained on.
#!
#! sbatch slurm_analyse_latent_space.sh

#SBATCH -J latent_space
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --mail-type=NONE

#! ── Data + checkpoint config — edit to match your experiment ─────────────────
#!
#! Model A: Gaussian only, turb 1/2/3
VAE_CHECKPOINT="checkpoints_vae_128_modelA/vae_oam_epoch100.pt"
MODES="gauss"
TURB_LEVELS="1 2 3"
OUTPUT_DIR="analysis_latent_space_modelA"
#!
#! Model B: all OAM modes, turb 3
#! VAE_CHECKPOINT="checkpoints_vae_128_modelB/vae_oam_epoch100.pt"
#! MODES="gauss p1 p2 p3 p4"
#! TURB_LEVELS="3"
#! OUTPUT_DIR="analysis_latent_space_modelB"
#!
#! Original VAE config: gauss + p4, all turb levels
#! VAE_CHECKPOINT="checkpoints_vae_128_orig/vae_oam_epoch100.pt"
#! MODES="gauss p4"
#! TURB_LEVELS=""        # leave empty = all levels
#! OUTPUT_DIR="analysis_latent_space_orig"
#!
#! ── Analysis settings ────────────────────────────────────────────────────────
MAT_PATH="/home/omo26/rds/hpc-work/MLMI4/DDPM/croped_2_2_pupil_data.mat"
N_PCA=6          # number of PCs to traverse (rows in the traversal grid)
N_STEPS=9        # decode steps per PC (columns), symmetric around mean
N_SIGMA=2.5      # range of traversal in standard deviations
TSNE_PERPLEXITY=30
TSNE_MAX_POINTS=2000   # subsample before t-SNE if dataset is larger
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

TURB_FLAG=""
if [ -n "$TURB_LEVELS" ]; then
    TURB_FLAG="--turb_levels $TURB_LEVELS"
fi

CMD="$PYTHON_EXEC -u analyse_latent_space.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --mat_path $MAT_PATH \
    --output_dir $OUTPUT_DIR \
    --modes $MODES \
    $TURB_FLAG \
    --n_pca $N_PCA \
    --n_steps $N_STEPS \
    --n_sigma $N_SIGMA \
    --tsne_perplexity $TSNE_PERPLEXITY \
    --tsne_max_points $TSNE_MAX_POINTS \
    --device cuda > logs/latent_space.$JOBID 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "VAE: $VAE_CHECKPOINT | modes=$MODES | turb_levels=$TURB_LEVELS"
echo "PCA: n_components=$N_PCA, steps=$N_STEPS, sigma=$N_SIGMA"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

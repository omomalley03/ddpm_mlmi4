#!/bin/bash
#!
#! SLURM job script: DDPM ablation (epsilon prediction, L_vlb objective, learned variance)
#!
#SBATCH -J latent_eps_lvlb_l
#SBATCH -A MLMI-dpc49-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

LATENT_PATH="data/celeba_latents.pt"

options="--mode train_latent --latent_path $LATENT_PATH --use_stable_diffusion_vae --total_steps 200000 --batch_size 128 --lr 2e-4 --save_dir checkpoints_latent/ablation_eps_lvlb_learned --save_every 50000 --log_every 1000 --prediction_target epsilon --objective_type l_vlb --variance_mode learned"

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PYTHON_EXEC="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/venv/bin/python"
application="$PYTHON_EXEC -u /rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/run.py"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddpm_mlmi4}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ablation_eps_lvlb_learned_${SLURM_JOB_ID}}"
export WANDB_DIR="${WANDB_DIR:-$workdir/wandb}"

options="$options --wandb --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME"

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs checkpoints_latent/ablation_eps_lvlb_learned "$WANDB_DIR"
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/latent_ablation_eps_lvlb_learned.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

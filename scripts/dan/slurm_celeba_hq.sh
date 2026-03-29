#!/bin/bash
#!
#! SLURM job script for DDPM training on CelebA-HQ on Wilkes3 (A100)
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J ddpm_celeba_hq
#! Which project should be charged:
#SBATCH -A MLMI-dpc49-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4):
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=36:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! ######################################################################################
#! Set training options here:
#! Note: CelebA-HQ is 256x256, so adjust batch_size and total_steps accordingly
#! Recommended: start with batch_size=8 on a single A100-80GB for this UNet config

options="--mode train --dataset celeba_hq --total_steps 500000 --batch_size 8 --lr 2e-4 --image_size 256 --save_dir checkpoints_celebahq --save_every 50000 --log_every 1000"

#! ######################################################################################

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application:
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

PYTHON_EXEC="/rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/venv/bin/python"
application="$PYTHON_EXEC -u /rds/user/dpc49/hpc-work/MLMI4/ddpm_mlmi4/run.py"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ddpm_mlmi4}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-celeba_hq_${SLURM_JOB_ID}}"
export WANDB_DIR="${WANDB_DIR:-$workdir/wandb}"

options="$options --wandb --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs checkpoints_celebahq "$WANDB_DIR"
JOBID=$SLURM_JOB_ID
CMD="$application $options > logs/out.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

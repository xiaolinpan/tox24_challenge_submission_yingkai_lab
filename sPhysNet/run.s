#! /bin/bash

#SBATCH --job-name=calc_input
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=log.out

module purge

#singularity exec --nv --overlay  /scratch/xp2042/overlay-v2/overlay-script-v3-15GB-500K.ext3:ro /home/xp2042/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u calc_inputs.py > log.o 2>&1"


singularity exec --nv --overlay  /scratch/xp2042/overlay-v2/overlay-script-v3-15GB-500K.ext3:ro /home/xp2042/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u train_sphysnet.py > train_log.o 2>&1"




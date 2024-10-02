#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q preemptable
#PBS -A argonne_tpc

export VENV_DIR=/lus/eagle/projects/argonne_tpc/chia-llama2/Megatron-DeepSpeed-SAM/venvs/2024-04-29
export VIRTUAL_ENV=$VENV_DIR

export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
module use /soft/modulefiles/
module load conda
conda activate base
source "${VENV_DIR}/bin/activate"

#export PYTHON_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/:$PYTHON_PATH

echo "loaded environment"

# Activate the runtime environment
#git config --global http.proxy http://proxy.alcf.anl.gov:3128

export PBS_O_WORKDIR=$(pwd)
cd "${PBS_O_WORKDIR}" || exit

echo $PBS_O_WORKDIR
bash ./train_aGPT_7B_chia.sh


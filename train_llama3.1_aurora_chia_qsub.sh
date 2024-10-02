#!/bin/bash --login
#PBS -l select=96,walltime=6:00:00,place=scatter
#PBS -A candle_aesp_CNDA
#PBS -l walltime=06:00:00
#PBS -N Llama3.1-8B-Test
#PBS -k doe
#PBS -q lustre_scaling


#####################################
# Llama3.1 Original Training Run
#
# Main production script for training
# Llama3.1-8B @ ALCF
#####################################

export PBS_O_WORKDIR="/flare/candle_aesp_CNDA/chia/Megatron-DeepSpeed-SAM"
export VENV_DIR=/flare/candle_aesp_CNDA/chia/Megatron-DeepSpeed-SAM/venvs/aurora_nre_models_frameworks-2024.2.1_u1
export VIRTUAL_ENV=$VENV_DIR

export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128

git config --global http.proxy http://proxy.alcf.anl.gov:3128

module use /soft/modulefiles
module load frameworks/2024.2.1_u1
module load mpich
source "${VENV_DIR}/bin/activate"

export HF_HOME="/flare/candle_aesp_CNDA/chia/.cache"
export HUGGING_FACE_HUB_TOKEN='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'
export PYTHONPATH=$PYTHONPATH:/home/chian/candle_aesp_chia/Megatron-DeepSpeed-SAM

# 1. Navigate into `$PBS_O_WORKDIR`
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
echo $HERE
# 2. source `ALCF/helpers.sh`
source "${HERE}/ALCF/helpers.sh" || exit
# 3. specify `DATA_FILE_LIST` for dolma dataset
export DATA_FILE_LIST="/flare/candle_aesp_CNDA/chia/Megatron-DeepSpeed-SAM/ALCF/data-lists/aurora/slimpajamaonly_list.txt" #change eventually
# 4. set custom output prefix - the place the converted model is
export CUSTOM_OUTPUT_PREFIX="ChiaTestAlpha1" #Doesn't do anything - mystery for later
#5 Set Checkpoint directory
export CKPT_DIR="/flare/candle_aesp_CNDA/chia/mds_checkpoints/llama-3.1-8b-pp2-tp1"
rm -rf ${CKPT_DIR}/.cache #untested, but did this manually
#6 Set Weights and Biases Project
export WB_PROJECT="aGPT-Team"
export WANDB_API_KEY="6be29223a2a4df6059287553241354e8cbaf122c"
export WANDB_ENTITY="aGPT"
#7. Print specifications & Other model params
export MICRO_BATCH=1
export NO_LOAD_OPTIM=1
export NO_LLAMA=1
#export OTHER_LLAMA=1
export SAVE_INTERVAL=10
#export TP=1
#export PP=2
# Set model configuration to match TinyLlama
export NLAYERS=32          # Number of layers (verify if this matches the checkpoint)
export HIDDEN=4096         # Update hidden size to 2048
export HEADS=32            # Update number of attention heads to match the checkpoint
export FFN_HIDDEN_SIZE=16384
export NUM_KV_HEAD=8
echo "$DATA_FILE_LIST"
echo "$CUSTOM_OUTPUT_PREFIX"
echo "$CKPT_DIR"
# 8. call setup from ALCF/helpers.sh
setup "$@" || exit
# 9. Set run_cmd
export run_cmd="${run_cmd}"
echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
# 10. Tell user where to find output
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
XPU_IGNORE_STRING="CCL_WARN|\ -\ INFO\ \-\ |real_accelerator\.py|numexpr\.utils|async_io|libaio"
# 11. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
echo "RUN: ${run_cmd}"
eval "${run_cmd}" |& grep -E -v "${XPU_IGNORE_STRING}" |& tee -a "${OUTPUT_LOG}"

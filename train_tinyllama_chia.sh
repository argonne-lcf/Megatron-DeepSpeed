#!/bin/bash --login

#####################################
# TinyLlama Original Training Run
#
# Main production script for training
# TinyLlama @ ALCF
#####################################

get_latest_iteration() {
    local ckpt_dir="$1"
    # Check if the 'latest' file exists
    if [ -f "${ckpt_dir}/latest" ]; then
        # Read the name of the latest checkpoint
        local latest_checkpoint=$(cat "${ckpt_dir}/latest")
        # Extract the iteration number from the checkpoint name
        local iter_number=$(echo "${latest_checkpoint}" | grep -oE '[0-9]+' | tail -1)
        echo "${iter_number}"
    else
        # No 'latest' file, assume iteration 0
        echo "0"
    fi
}

# 1. Navigate into `$PBS_O_WORKDIR`
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
echo $HERE
# 2. source `ALCF/helpers.sh`
source "${HERE}/ALCF/helpers.sh" || exit
# 3. specify `DATA_FILE_LIST` for dolma dataset
export DATA_FILE_LIST="/flare/candle_aesp_CNDA/chia/Megatron-DeepSpeed-SAM/ALCF/data-lists/aurora/slimpajamaonly_list.txt" #change eventually
# 4. set custom output prefix - the place the converted model is
export CUSTOM_OUTPUT_PREFIX="ChiaTestAlpha1"
#5 Set Checkpoint directory
export CKPT_DIR="/flare/candle_aesp_CNDA/chia/mds_checkpoints/TinyLlama_v1.1_MDS"
#7. Print specifications & Other model params
export MICRO_BATCH=4
export NO_LOAD_OPTIM=1
#export PP=24
# Set model configuration to match TinyLlama
export NLAYERS=22          # Number of layers (verify if this matches the checkpoint)
export HIDDEN=2048         # Update hidden size to 2048
export HEADS=32            # Update number of attention heads to match the checkpoint
export FFN_HIDDEN_SIZE=5632
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

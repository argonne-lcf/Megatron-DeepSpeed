#!/bin/bash --login

#####################################
# AuroraGPT-7B
#
# Main production script for training
# AuroraGPT-7B @ ALCF
#####################################


# 1. Navigate into `$PBS_O_WORKDIR`
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
# 2. source `ALCF/helpers.sh`
source "${HERE}/ALCF/helpers.sh" || exit
# 3. specify `DATA_FILE_LIST` for dolma dataset
export DATA_FILE_LIST="/lus/eagle/projects/argonne_tpc/chia-llama2/Megatron-DeepSpeed-SAM/ALCF/data-lists/polaris/dolma_w_biodata.txt" #change eventually
# 4. set custom output prefix - the place the converted model is
export CUSTOM_OUTPUT_PREFIX=""
echo "$DATA_FILE_LIST"
echo "$CUSTOM_OUTPUT_PREFIX"
# 5. call setup from ALCF/helpers.sh
setup "$@" || exit
# 6. Set run_cmd
export run_cmd="${run_cmd}"
echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
# 7. Tell user where to find output
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
XPU_IGNORE_STRING="CCL_WARN|\ -\ INFO\ \-\ |real_accelerator\.py|numexpr\.utils|async_io|libaio"
# 8. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
eval "${run_cmd}" |& grep -E -v "${XPU_IGNORE_STRING}" |& tee -a "${OUTPUT_LOG}"

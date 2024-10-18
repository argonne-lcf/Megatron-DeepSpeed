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

# 3. call `setup` from `./ALCF/helpers.sh`
setup "$@" || exit
# export run_cmd="${run_cmd}"
echo "${run_cmd[@]}" | tee -a "${OUTPUT_LOG}"

# 4. Tell user where to find output
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"

# # 5. Ignore the following strings on Intel XPU devices
# #    (otherwise they'll clutter up logs)
# XPU_IGNORE_STRING="CCL_WARN|\ -\ INFO\ \-\ |real_accelerator\.py|numexpr\.utils|async_io|libaio"

# if [[ $(ezpz_get_machine_name) == "aurora" ]]; then
#     module unload mpich && module load mpich
# fi
#
# 6. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
# eval "${run_cmd[@]}" |& tee -a "${OUTPUT_LOG}"
eval "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"

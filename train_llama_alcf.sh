#!/bin/bash --login

###############################################################################
# Check if running in DEBUG=1 mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
###############################################################################
if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

###############################################################################
# Print (but DO NOT EXECUTE !!) each command that would be ran.
#
# Enable with: NOOP=1 PBS_O_WORKDIR=$(pwd) bash train_llama_alcf.sh
###############################################################################
if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

#####################
# MAIN PROGRAM LOGIC
#####################
main() {
    # 1. Navigate into `$PBS_O_WORKDIR`
    cd "${PBS_O_WORKDIR}" || exit
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    # 2. source `ALCF/helpers.sh`
    source "${HERE}/ALCF/helpers.sh" || exit
    # 3. call `setup` from `./ALCF/helpers.sh`
    setup || exit
    # 4. Take custom args
    export custom_args=" $@"
    # 5. Update ${run_cmd} (from setup ALCF/helpers.sh) with ${custom_args}
    export run_cmd="${run_cmd} ${custom_args}"
    # 6. Add "${run_cmd}" to output log
    echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
    # 7. Tell user where to find output
    printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
    # 8. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
    eval "${run_cmd}" |& tee -a "${OUTPUT_LOG}"
    set +x
}

main

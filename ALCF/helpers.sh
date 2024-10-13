#!/bin/bash --login
###############################################################################
# [`ALCF/helpers.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh)
#
# Contains helper functions for launching `../train_llama_alcf.sh`
#
# To use, on any of {Polaris, Aurora, Sunspot} @ ALCF:
#
#     ```bash
#     $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
#     $ cd Megatron-DeepSpeed
#     $ export PBS_O_WORKDIR=$(pwd) && source ALCF/helpers.sh && ezpz_setup
#     ```
#
# and this will, automatically:
#
# 1. Setup python (conda + virtual environment)
#
# 2. Parse `$PBS_*` env vars to build appropriate `alias launch='mpiexec ...'`
#    command for launching across all GPUs in our active PBS job.
###############################################################################

##################
# helpers_main
#
# This will get called automatically when running:
#
# ```bash
# $ cd Megatron-DeepSpeed
# $ PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh
# ```
#
# - This will set `"${WORKING_DIR}"`, according to:
#
#       1. if `${PBS_O_WORKDIR}` is nonzero, use this
#       2. else, if `${SLURM_SUBMIT_DIR}` is nonzero use this
#       3. else, use `$(pwd)`
#
#   this is _crucial_ since many of the functions below use paths
#   which are defined relative to this "${WORKING_DIR}"
#   (e.g. virtual environment, location of executables, etc.)
##################
helpers_main() {
    # NOTE: for debug mode, run with `DEBUG=1`
    if [[ -n "${DEBUG:-}" ]]; then
        set -euxo
    fi
    if [[ -n "${PBS_O_WORKDIR}" ]]; then
        WORKING_DIR="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
        WORKING_DIR="${SLURM_SUBMIT_DIR}"
    else
        echo "Unable to detect PBS or SLURM working directory info..."
        WORKING_DIR=$(python3 -c 'import os; print(os.getcwd())')
        echo "Using ${WORKING_DIR} as working directory..."
    fi
    export WORKING_DIR="${WORKING_DIR}"
    printf "Using WORKING_DIR: %s\n" "${WORKING_DIR}"
}

##############################################################################
# setup
#
# All-in-one helper function.
#
# - Explicitly, this will:
#    - Identify the machine we're on
#
#    - Setup `python`
#       1. Load `conda`
#       2. Setup `venv` on top of `conda`
#
#    - Ensure all dependencies are installed
#
#    - Clone + Install [`saforem2/ezpz`](https://github.com/saforem2/ezpz)
#       - Source [`ezpz/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
#           - This provides `{ezpz_setup_python, ezpz_setup_job}` (called below)
#
#    - Set runtime options
#
#    - Build `deepspeed_config.json`
#
#    - Build {logs, checkpoints, etc} dirs, named according to specifics of
#       current run
#
#    - Specify additional `deepspeed` arguments
#
#    - Ensure executable exists at expected path
#
#    - Setup data + tokenizer via `TOKENIZER_TYPE`
#
#    - Print job info
#
#    - Save `.env` to `CKPT_DIR` for safe keeping
#
#    - Check that we're not already running, and if so, exit.
#
#    - Setup run command to be executed.
##############################################################################
setup() {
    # Identify machine we're on
    get_machine || exit
    ##########################################################################
    # ezpz_setup will:
    # 1. Setup python
    #     - load base conda
    #     - (if necessary) create virtual environment on top of base conda
    #     - activate virtual environment from ^
    # 2. Install ezpz (if needed)
    # 3. Parse PBS_* environment variables to determine:
    #     - NHOSTS (by counting number of lines in $PBS_NODEFILE)
    #     - NGPU_PER_HOST (by magic)
    #     - NGPUS (= NHOSTS * NGPU_PER_HOST)
    # 4. Use these (^) to build our launch command
    ezpz_setup || exit
    ##########################################################################
    install_dependencies
    # Set command line arguments to pass to `"${EXEC}"`
    setParams || exit
    # Create `deepspeed_config.json` from runtime params from ^
    buildDSconfig || exit
    # Specify output directory for {logs, checkpoints, etc.}
    setOutput || exit
    # Specify additional `deepspeed` arguments (dependent on _newly created_ variables)
    set_args || exit
    # Ensure executable exists in expected path
    check_executable "${EXEC:-${WORKING_DIR}/pretrain_gpt_alcf.py}"
    dfl="${DATA_FILE_LIST:-}"
    # Setup data + tokenizer via `DATA_FILE_LIST` and `TOKENIZER_TYPE`
    tok="${TOKENIZER_TYPE:-Llama2}"
    setup_tokenizer_and_data "${tok}" "${dfl}" || exit
    make_data || exit
    # Print job info
    printJobInfo || exit
    # Save `.env` to `CKPT_DIR` for safe keeping
    save_dotenv "${CKPT_DIR}" || exit
    # Check that were not already running, if so, exit.
    check_and_kill_if_running || exit
    # Setup run command to be executed
    setup_run_cmd "$@" || exit
}

#####################################################
# setup_run_cmd
#
# Build run command to be executed.
#####################################################
setup_run_cmd() {
    ##############################
    # take in additional arguments
    # and append them directly to
    # the end of the `run_cmd`
    custom_args="$@"
    ##############################
    #### Make it easy to track experiments by date ###################
    year="$(date "+%Y")"
    month="$(date "+%m")"
    day="$(date "+%Y-%m-%d")"
    today="$(date "+%Y-%m-%d")" # kept for backwards compatibility
    started_at="$(date "+%Y-%m-%d-%H%M%S")"
    export YEAR="${year}"
    export MONTH="${month}"
    export DAY="${day}"
    export TODAY="${today}"
    export STARTED_AT="${started_at}"
    ##################################################################
    # NOTE: to launch with DeepSpeed instead of mpiexec:
    # `export LAUNCH_WITH=deepspeeed && bash train_llama_alcf.sh`
    ##################################################################
    setupLauncher "${LAUNCH_WITH:-MPICH}" || exit
    TBDIR="${CKPT_DIR}/tensorboard"
    mkdir -p "${TBDIR}"
    export data_cache_path="${CKPT_DIR}/${DATA_CACHE_PATH}" && mkdir -p "${data_cache_path}"
    printf "\n"
    echo "Using data_cache_path: ${data_cache_path}"
    export DEFAULTS="\
        --split 100,0,0 \
        --log-interval 1 \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-masked-softmax-fusion \
        --no-gradient-accumulation-fusion \
        --accumulate-allreduce-grads-in-fp32 \
        --log-timers-to-tensorboard \
        --log-optimizer-states-to-tensorboard"
    OVERRIDE_CKPT_OPT_PARAM="${OVERRIDE_CKPT_OPT_PARAM:-}"
    if [[ -z "${OVERRIDE_CKPT_OPT_PARAM:-}" ]]; then
        DEFAULTS="${DEFAULTS} --use-checkpoint-opt_param-scheduler"
    fi
    if [[ "${SP}" -ge 2 ]]; then
        export DEFAULTS="${DEFAULTS} --ds-sequence-parallel-size ${SP} --force-ds-sequence-parallel"
    fi
    ##################################################################
    # WARN: to disable Llama-type architectures, toggle via:
    # `NO_LLAMA=1 bash train_llama_alcf.sh`
    ##################################################################
    if [[ -z "${NO_LLAMA:-}" ]]; then
        llama_flags="${LLAMA_ARGS}\
            --num-key-value-heads ${NUM_KV_HEAD} \
            --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
            "
    else
        echo "!! Running in NO_LLAMA MODE !!"
        llama_flags=""
    fi
    export run_cmd="
        ${LAUNCHER} \
        --${DTYPE} \
        ${DEFAULTS} \
        --optimizer ${OPT} \
        --adam-beta1=${ADAM_BETA1} \
        --adam-beta2=${ADAM_BETA2} \
        --adam-eps=${ADAM_EPS} \
        --weight-decay=${WEIGHT_DECAY} \
        --save ${CKPT_DIR} \
        --load ${CKPT_DIR} \
        --seq-length ${SEQ} \
        --num-layers ${NLAYERS} \
        --hidden-size ${HIDDEN} \
        --tensorboard-dir ${TBDIR} \
        --train-iters ${TRAIN_ITERS} \
        --eval-iters ${EVAL_ITERS} \
        --distributed-backend ${BE} \
        --num-attention-heads ${HEADS} \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --max-position-embeddings ${SEQ} \
        --micro-batch-size ${MICRO_BATCH} \
        --tensor-model-parallel-size ${TP} \
        --global-batch-size ${GLOBAL_BATCH} \
        --pipeline-model-parallel-size ${PP} \
        --data-cache-path ${data_cache_path} \
        ${DATA_FLAGS} \
        ${LR_ARGS} \
        ${llama_flags} \
        ${FLASH_ARG} \
        ${TIMING_STR} \
        ${TOKENIZER_FLAGS} \
        ${ds_args} \
        ${gpt_args[*]} \
        ${custom_args}
        "
}

save_dotenv() {
    if [[ "$#" -ne 1 ]]; then
        estr="[error]"
        printf "%s Expected one argument (outdir). Received: %s" "$(printRed "${estr}")" "$#"
    else
        outdir="$1"
        mkdir -p "${outdir}"
        module list
        dotenv_file="${outdir}/.env"
        echo "Saving environment to ${dotenv_file}"
        printenv | grep -v "LS_COLORS" >"${dotenv_file}"
        export DOTENV_FILE="${dotenv_file}"
    fi
}

######################################################################
# get_machine_name:
#
# Return current machine name, as lowercase string
#
# Example:
#   ```bash
#   $ machine_name=$(get_machine_name)
#   $ echo "machine_name: ${machine_name}"
#   ```
######################################################################
get_machine_name() {
    if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == x3* || $(hostname) == polaris* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == nid* ]]; then
        machine="perlmutter"
    else
        machine=$(hostname)
    fi
    echo "${machine}"
}

get_machine() {
    if [[ $(hostname) == x4* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == x3* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == nid* ]]; then
        machine="perlmutter"
    else
        echo "Unknown MACHINE. Setting MACHINE to $(hostname) and continuing..."
    fi
    export MACHINE="${machine}"
    printf "Running on: %s\n" "$(printBlue "${MACHINE}")"
}

check_and_kill_if_running() {
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]]; then
        echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}"
    else
        echo "Not currently running. Continuing!"
    fi
}

setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$((NHOSTS * NGPU_PER_HOST))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}

printJobInfo() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "- MPICH_DIR=${MPICH_DIR:-${MPI_ROOT}}"
    echo "- Using $(which python3)"
    echo "- WORLD_SIZE:${WORLD_SIZE-}"
    echo "- BACKEND: ${BE:-}"
    echo "- MODEL_TYPE: ${MODEL_TYPE:-}"
    echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST:-}"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
}

#############################################################################
# setupLauncher: Launch with one of `{mpiexec, deepspeed}`.
#
# Explicitly, look for `LAUNCH_CMD` in environment and launch accordingly.
# Will use `mpiexec` by default.
# To launch with `deepspeed` instead, specify `LAUNCH_CMD=deepspeed`, e.g.
#
#     ```bash
#     PBS_O_WORKDIR=$(pwd) LAUNCH_CMD=deepspeed bash train_llama_alcf.sh
#     ```
#
# will launch with `deepspeed` instead of `mpiexec`.
#############################################################################
setupLauncher() {
    if [[ "$#" == 1 ]]; then
        local dist_launcher="$1"
    else
        local dist_launcher="${LAUNCH_WITH:-${LAUNCH_CMD:-"MPICH"}}"
    fi
    if [[ "${dist_launcher}" == "deepspeed" ]]; then
        # Save {PATH, LD_LIBRARY_PATH, ...} to .deepspeed_env
        saveDSenv || exit
        # Assert `./hostfile_deepspeed` exists
        export hfds="${WORKING_DIR}/hostfile_deepspeed"
        make_ds_hostfile || exit
        export LAUNCHER="deepspeed --hostfile $hfds --launcher MPICH ${EXEC}"
    else
        if [ -n "$SOPHIA" ]; then
            # LAUNCHER="${DIST_LAUNCH} --pmi=pmix --genvall $(which python3) -Wignore ${EXEC}"
            # dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
            LAUNCHER="mpiexec --verbose $(env | grep -v -e '"' -e ';' -e '*' -e '-' -e '__' -e '}' -e '|' -e ')' -e esac -e else -e fi| awk -F= '{print "-x", $1}' | tr '\n' ' ') \
                        -n ${num_gpus} --hostfile ${hostfile} $(which python3) -Wignore ${EXEC}"
            echo "${LAUNCHER}"
            export LAUNCHER="${LAUNCHER}"
        elif [[ -n "${DIST_LAUNCH}" ]]; then
            mn=$(get_machine_name)
            if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
                LAUNCHER="${DIST_LAUNCH} --pmi=pmix --genvall $(which python3) -Wignore ${EXEC}"
            else
                LAUNCHER="${DIST_LAUNCH} --genvall $(which python3) -Wignore ${EXEC}"
            fi
            export LAUNCHER="${LAUNCHER}"
        else
            echo "[setupLauncher][INFO]: Saving environment to: .env-${PBS_JOBID}"
            printenv | tee ".env-${PBS_JOBID}"
            echo "[setupLauncher][ERROR]: DIST_LAUNCH not found in environment !!"
        fi
    fi
    printf "Launching with: %s\n" "$(printRed "${dist_launcher}")"
    printf " %s" "$(printMagenta "${LAUNCHER}")"
}

set_lr_args() {
    LR_ARGS="--lr ${LR} --lr-decay-style cosine"
    if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
        LR_ARGS="${LR_ARGS} --lr-decay-iters ${LR_DECAY_ITERS}"
    fi
    if [[ -n "${LR_WARMUP_FRAC}" ]]; then
        LR_ARGS="${LR_ARGS} --lr-warmup-fraction ${LR_WARMUP_FRAC}"
    fi
    echo "LR_ARGS: ${LR_ARGS}"
    export LR_ARGS="${LR_ARGS}"
}

#########################################################################
# `get_batch_size_on_polaris`: Identify MICRO_BATCH to use on Polaris.
#
# - In particular, it seems that different node counts allow for different
#   `MICRO_BATCH` sizes.
#
#   Explicitly:
#
#       - [1 <= NHOSTS <= 2]: `MICRO_BATCH=1`
#       - [3 <= NHOSTS <= 7]: `MICRO_BATCH=2`
#       - [8 <= NHOSTS]:      `MICRO_BATCH=4`
#
#   are the largest batch sizes that fit in memory at various node counts.
#########################################################################
get_batch_size_on_polaris() {
    if [[ $(hostname) == x3* ]]; then
        nhosts=$(wc -l <"${HOSTFILE:-${PBS_NODEFILE}}")
        if [[ "${nhosts}" == 1 || "${nhosts}" == 2 ]]; then
            mbs=1
        elif [[ "${nhosts}" -ge 3 && "${nhosts}" -le 7 ]]; then
            mbs=2
        elif [[ "${nhosts}" -ge 8 ]]; then
            mbs=4
        fi
    fi
    echo "${mbs}"
}

_get_num_hosts_from_hostfile() {
    if [[ "$#" == 1 ]]; then
        if [[ -f "$1" ]]; then
            nhosts=$(wc -l <"$1")
            echo "${nhosts}"
        else
            exit 1
        fi
    else
        exit 1
    fi
}

###########################################
# get_grad_acc_steps_on_aurora
#
# NOTE:
# We use different numbers of gradient
# accumulation steps (GAS) depending
# on the number of hosts in our job.
#
# Each host has:
#
#   [2 tiles] x [6 xpus / tile] = 12 xpus
#
# |    nnhosts    |   nhosts  |  GAS  |
# |:-------------:|:---------:|:-----:|
# | 64 <= n < inf | [64, inf) |   1   |
# | 32 <= n < 64  | [32, 64)  |   2   |
# | 16 <= n < 32  | [16, 32)  |   4   |
# |  0 <= n < 16  | [0, 16)   |   8   |
###########################################
get_grad_acc_steps_on_aurora() {
    if [[ "$#" == 0 ]]; then
        hf="${HOSTFILE:-${PBS_NODEFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}}"
    elif [[ "$#" == 1 ]]; then
        hf="$1"
    else
        echo "Expected exactly 0 or 1 arguments, received: $#"
        exit 1
    fi
    nhosts=$(wc -l <"${hf}")
    if [[ 64 -le "${nhosts}" ]]; then
        gas=1
    elif [[ 32 -le "${nhosts}" && "${nhosts}" -lt 64 ]]; then
        gas=2
    elif [[ 16 -le "${nhosts}" && "${nhosts}" -lt 32 ]]; then
        gas=4
    else
        gas=8
    fi
    echo "${gas}"
}

set_ccl_vars_on_aurora() {
    export CCL_KVS_MODE=mpi
    export CCL_CONFIGURATION_PATH=""
    export CCL_CONFIGURATION=cpu_gpu_dpcpp
    # export CCL_ROOT=/tmp/oneccl/
    # export LD_LIBRARY_PATH=${CCL_ROOT}/lib:$LD_LIBRARY_PATH
    # export CPATH=${CCL_ROOT}/include:$CPATH
    # export LIBRARY_PATH=${CCL_ROOT}/lib:$LIBRARY_PATH
    export CCL_KVS_CONNECTION_TIMEOUT=3600
    export FI_CXI_RX_MATCH_MODE=hybrid
    export CCL_BCAST=double_tree

    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export CCL_PROCESS_LAUNCHER=pmix # Required by Aurora mpich
    export FI_PROVIDER=cxi           # Required by Aurora mpich
    export PALS_PMI=pmix             # Required by Aurora mpich
    export CCL_ATL_TRANSPORT=mpi     # Required by Aurora mpich
    export TORCH_LLM_ALLREDUCE=1
    export CCL_SYCL_ESIMD=1
    export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0 # Required by current oneCCL (MLSL-2881)
    export CCL_ENABLE_SYCL_KERNELS=1
    export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
    export FI_CXI_DEFAULT_CQ_SIZE=1048576
    export FI_CXI_RX_MATCH_MODE=hybrid
    export CCL_BCAST=double_tree
}

##############################################################################
# setParams
#
# Set / configure run options by parsing environment.
#
# - any of the declared options below can be overridden
#     dynamically at runtime, e.g. to run with a `MICRO_BATCH` size of 2:
#         ```bash
#         $ PBS_O_WORKDIR=$(pwd) MICRO_BATCH=2 bash train_llama_alcf.sh
#         ```
##############################################################################
setParams() {
    FLASH_ARG=""
    LLAMA_ARGS="--attention-dropout 0 --hidden-dropout 0"
    # ---- [Parallelism Settings] -------------------------------------------+
    # ------ [Aurora] -------||------ [SunSpot] -------------
    # if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
    mn=$(get_machine_name)
    if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
        TP=${TP:-1} # TP = 1
        export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
        export CCL=${CCL:-ccl}      # CCL
        export BE="${CCL}"          # COMMUNICATION BACKEND = CCL
        export DTYPE=${DTYPE:-bf16} # DTYPE: bf16
        # export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}     # GRADIENT_ACC_STEPS
        gas=$(get_grad_acc_steps_on_aurora "${PBS_NODEFILE:-${HOSTFILE:-${hostfile}}}")
        export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-${gas}}"
        # export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-$(get_grad_acc_steps_on_aurora "$@)}"
        echo "[setParams] Using GRAD_ACC_STEPS: ${GRAD_ACC_STEPS}"
        MICRO_BATCH=${MICRO_BATCH:-4} # MICRO_BATCH = 4
        #### [sam: 08/17/2024] ##########################################
        # Use best set of CCL env vars from Gordon Bell runs on Aurora
        set_ccl_vars_on_aurora
        #################################################################
        #### [sam: 06/20/2024] ###############################################
        # export CCL_PROCESS_LAUNCHER=pmix
        # export CCL_ATL_TRANSPORT=mpi
        # !XXX: USE KEY VALUE STORE FIX ON AURORA [2024-06-20]
        # use_kvs_fix_on_aurora  # <-- why are these different from those in update_ccl_env_vars_aurora ??
        # update_ccl_env_vars_aurora
        ######################################################################
        if [[ -z "${USE_FLASH_ATTN:-}" ]]; then
            # NOTE: if NO_FLASH_ATTN is NON-empty; then NO FLASH ATTN !!
            export NO_FLASH_ATTN=1 # disabled on [2024-06-20] waiting on fix...
            if [[ -n "${NO_FLASH_ATTN-}" ]]; then
                echo "Not using flash-attn!!"
            else
                FLASH_ARG="--use-flash-attn-builder"
            fi
        else
            echo "Using flash-attn !!"
            FLASH_ARG="--use-flash-attn-builder"
        fi
        ######################################################################
    # +--------[Polaris]-----------------------------------+
    # elif [[ $(hostname) == x3* ]]; then
    elif [[ "${mn}" == "polaris" || "${mn}" == "sirius" || -n "$SOPHIA" ]]; then
        # export LAUNCH_CMD="${LAUNCH_CMD:-deepspeed}"
        TP=${TP:-1}               # TP = 2
        export NCCL=${NCCL:-nccl} # NCCL
        export BE="${NCCL}"       # BE = NCCL
        # export DTYPE=${DTYPE:-bf16}                   # DTYPE: BF16 ??
        export DTYPE=${DTYPE:-fp16}                # DTYPE: FP16
        export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8} # GRADIENT_ACC_STEPS
        # NOTE: MICRO_BATCH is exported below
        # MICRO_BATCH=${MICRO_BATCH:-2}    # MICRO_BATCH = 8
        export MICRO_BATCH="${MICRO_BATCH:-$(get_batch_size_on_polaris)}"
        if [[ -n "${NO_FLASH_ATTN-}" ]]; then
            echo "Not using flash-attn!!"
        else
            FLASH_ARG="--use-flash-attn-v2"
        fi
        echo "Setting up AWS NCCL OFI Plugin on Polaris..."
        source "${WORKING_DIR}/ALCF/aws_ofi_nccl_plugin.sh" || exit
    # +--------[Perlmutter]---------------------------------+
    # elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
    elif [[ "${mn}" == login* || "${mn}" == nid* ]]; then
        TP="${TP:-2}"
        export NCCL="${NCCL:-nccl}"
        export BE="${NCCL}"
        export DTYPE="${DTYPE:-bf16}"
        MICRO_BATCH="${MICRO_BATCH:-8}"
        if [[ -n "${NO_FLASH_ATTN-}" ]]; then
            echo "Not using flash-attn!!"
        else
            FLASH_ARG="--use-flash-attn-v2"
        fi
    fi
    # +----------------------------------------------------------------------+
    export TP="${TP:-1}"
    export PP="${PP:-1}"
    export SP="${SP:-1}"
    export FLASH_ARG="${FLASH_ARG}"
    export DTYPE="${DTYPE:-bf16}"
    export OPT="${OPT:-adamw}"
    export ADAM_BETA1="${ADAM_BETA1:-0.9}"
    export ADAM_BETA2="${ADAM_BETA2:-0.95}"
    export ADAM_EPS="${ADAM_EPS:-0.00001}" # 1 * 10^{-5}
    export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
    export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
    NHOSTS=$(wc -l <"${HOSTFILE}")
    if [[ -z "${NGPU_PER_HOST:-}" ]]; then
        NGPU_PER_HOST=$(python3 -c 'import ezpz as ez; print(ez.get_gpus_per_node())')
    fi
    export NGPU_PER_HOST="${NGPU_PER_HOST}"
    export WORLD_SIZE="${WORLD_SIZE:-$((NHOSTS * NGPU_PER_HOST))}"
    # +---[Llama2 7B Config]--------------------------------------------------+
    # export MODEL_KEY="Llama-7B"
    export HEADS=${HEADS:-${NHEADS:-32}}             # NUMBER OF ATEN HEADS
    export NLAYERS=${NLAYERS:-${NUM_LAYERS:-32}}     # NUMBER OF LAYERS
    export HIDDEN=${HIDDEN:-4096}                    # HIDDEN SIZE
    export NUM_KV_HEAD=${NUM_KV_HEAD:-8}             # GROUP ATTENTION
    export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008} # FFN HIDDEN SIZE
    # +---[Run Settings]------------------------------------------------------+
    export SEQ=${SEQ:-4096}                                                               # SEQ_LEN: 4096
    export ZERO_STAGE=${ZERO_STAGE:-1}                                                    # ZERO OFFLOADING STAGE
    export MICRO_BATCH=${MICRO_BATCH:-8}                                                  # MICRO BATCH SIZE
    export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}                                            # GRADIENT ACCUMULATION STEPS
    export EVAL_ITERS="${EVAL_ITERS:-10}"                                                 # NUMBER OF EVAL ITERS TO RUN
    export EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"                                        # HOW FREQUENTLY TO RUN EVAL
    export SAVE_INTERVAL=${SAVE_INTERVAL:-50}                                             # HOW FREQUENTLY TO SAVE CKPTS
    export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"                                      # TIMING VERBOSITY IN LOGS
    export ACT_CKPT_NUM_LAYERS="${ACT_CKPT_NUM_LAYERS:-1}"                                # NUM LAYERS TO CHECKPOINT ACTIVATIONS
    export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}                # USE ACTIVATION CHECKPOINTING ?
    export GLOBAL_BATCH_MAX=$((WORLD_SIZE * MICRO_BATCH * GRAD_ACC_STEPS / TP / PP / SP)) # MAX GLOBAL BATCH SIZE
    export GLOBAL_BATCH="${GLOBAL_BATCH:-${GLOBAL_BATCH_MAX}}"                            # WILL USE MAX IF NOT SET IN ENVIRONMENT
    # export TRAIN_ITER=${TRAIN_ITER:-317892}             # NUMBER OF TRAIN ITERS
    if [[ -z "${TRAIN_ITERS:-${TRAIN_ITER:-}}" ]]; then
        export TRAIN_TOKENS=${TRAIN_TOKENS:-2000000000000}
        export TRAIN_ITERS=$((TRAIN_TOKENS / SEQ / GLOBAL_BATCH))
        printf "TRAIN_TOKENS=%s (=%sB tokens)\n" "${TRAIN_TOKENS}" "$((TRAIN_TOKENS / 10 ** 9))"
        printf "TRAIN_ITERS=%s\n" "${TRAIN_ITERS}"
    else
        export TRAIN_ITERS="${TRAIN_ITERS:-${TRAIN_ITER:-}}"
    fi
    export MODEL_TYPE="llama-gb${GLOBAL_BATCH}-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden" # STRING FOR IDENTIFYING MODEL
    # NOTE: [2024-07-10] #####################################################
    # - [sam]: For whatever reason, it seems that using
    #   sequence-parallelism (SP) > 1 is INCOMPATIBLE with
    #   rotary-position-embeddings (ROPE).
    #
    #   For this reason, we only use the default LLAMA_ARGS when SP=0.
    ##########################################################################
    if [[ "${SP}" == 1 ]]; then
        export LLAMA_ARGS="${LLAMA_ARGS} --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
    else
        export LLAMA_ARGS=""
        echo "NOT USING ROTARY EMBEDDINGS! LLAMA_ARGS=${LLAMA_ARGS}"
    fi
    # -----[Learning Rate Settings]--------------------------------------------
    export LR=${LR:-0.0003}                       # LEARNING_RATE
    export LR_WARMUP_FRAC=${LR_WARMUP_FRAC:-0.05} # LEARNING RATE WARMUP
    export LR_DECAY_ITERS=${LR_DECAY_ITERS:-}     # LR DECAY ITERS
    set_lr_args
    # -----[Learning Rate Settings]--------------------------------------------
    if [[ "${TIMING_LOG_LEVEL}" -ge 1 ]]; then
        TIMING_STR="\
            --timing-log-level ${TIMING_LOG_LEVEL} \
            --log-timers-to-tensorboard \
            --log-optimizer-states-to-tensorboard \
        "
    else
        TIMING_STR=""
    fi
}

##############################################
# set_args
#
# Specify additional (DeepSpeed specific)
# arguments to pass to pretrain_gpt_alcf.py
##############################################
set_args() {
    # ---- Set DeepSpeed arguments --------------------------------
    ds_args=" "
    ds_args=" --deepspeed ${ds_args}"
    if [[ $PP == 1 ]]; then
        ds_args=" --no-pipeline-parallel ${ds_args}"
    fi
    ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
    ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
    if [[ "${ZERO_STAGE}" == 3 ]]; then
        ds_args="--use-mics ${ds_args}"
    fi
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
        # --checkpoint-activations \
        # --deepspeed-activation-checkpointing
    fi
    export ds_args
    # ---------------------------------------------------------------
    gpt_args=()
    # we are now using activation checkpoint provided by megatron, see below.
    # ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        gpt_args+=(
            "--checkpoint-activations"
            "--checkpoint-num-layers ${ACT_CKPT_NUM_LAYERS}"
        )
    fi
    export gpt_args
}

make_ds_hostfile() {
    export GPUS_PER_NODE="${GPUS_PER_NODE:-${NGPU_PER_HOST:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}}"
    # ---- Make MPICH hostfile ----------------
    hf="${HOSTFILE:-${PBS_NODEFILE}}"
    export hostfile_mpich=hostfile_mpich
    cat "${hf}" >"${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "${hf}" >"${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}

###########################################
# ezpz_setup
#
# 1. Clone [`saforem2/ezpz`](https://github.com/saforem2/ezpz) (if necessary)
#    to `"${WORKING_DIR}/deps/ezpz/"`
#
# 2. Source [`ezpz/src/ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
#    - This provides `{ezpz_setup_python, ezpz_setup_job}` (called below)
#
# 3. Call `ezpz_setup_python` (from `ezpz/bin/utils.sh`):
#    - This will setup conda + virtual enviroment
#
# 4. Call `ezpz_setup_job` (from `ezpz/bin/utils.sh`):
#    - This will parse `$PBS_*` variables and build launch cmd
#
# 3. Call `_ezpz_install` (from `Megatron-DeepSpeed/ALCF/helpers.sh`):
#    - Install ezpz from `"${WORKING_DIR}/depz/ezpz/"`
###########################################
ezpz_setup() {
    # setup_alcf "$@"
    # file=$(mktemp)
    # curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/getjobenv > "${file}"
    # shellcheck source=../deps/ezpz/src/ezpz/bin/utils.sh
    ezdir="${WORKING_DIR}/deps/ezpz"
    if [[ -d "${ezdir}" ]]; then
        echo "Found ezpz in ${ezdir}"
    else
        mkdir -p "$(dirname "${ezdir}")"
        git clone https://github.com/saforem2/ezpz "${ezdir}"
    fi
    # shellcheck source=../deps/ezpz/src/ezpz/bin/utils.sh
    source "${ezdir}/src/ezpz/bin/utils.sh" || exit
    ezpz_setup_python
    ezpz_setup_job "$@"
    ezpz_pip_loc=$(python3 -m pip list | grep ezpz | awk '{print $NF}')
    if [[ -z "${ezpz_pip_loc:-}" ]]; then
        printf "[ezpz_install] Installing ezpz from %s\n" "${ezdir}"
        python3 -m pip install -e "${ezdir}" --require-virtualenv
    else
        printf "[ezpz_install] Found ezpz @ %s\n" "${ezpz_pip_loc}"
    fi
}

#######################################################################
# ezpz_test: Run simple test to make sure all nodes in working order
#######################################################################
ezpz_test() {
    printf "%s" "[$(printBlue 'ezpz:test_dist')][INFO] Running ezpz.test_dist...\n"
    # [ -n "${PBS_O_WORKIR}" ] && ezpz_savejobenv || ezpz_getjobenv
    # python3 -Wignore -m ezpz.jobs && source "${PBS_O_WORKDIR}/.jobenv"
    printf "%s" "[$(printBlue 'ezpz:test_dist')] Running test: ${eztest}\n"
    eztest="TRAIN_ITERS=50 ${LAUNCH_CMD} python3 -Wignore -m ezpz.test_dist"
    eval "${eztest}"
    printf "%s" "[$(printBlue 'ezpz:test_dist')] Done with test!\n"
}

############################################################################
# saveDSenv
#
# Save important environment variables to .deepspeed_env, which will be
# forwarded to ALL ranks with DeepSpeed
############################################################################
saveDSenv() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        echo "http_proxy=${http_proxy:-}"
        echo "https_proxy=${https_proxy:-}"
        echo "CFLAGS=${CFLAGS}"
        echo "PYTHONUSERBASE=$PYTHONUSERBASE"
    } >.deepspeed_env
}

get_output_prefix() {
    # ---- Specify output location --------------------------------
    pre="ws${WORLD_SIZE}_ds_stage${ZERO_STAGE}_nl${NLAYERS}"
    pre="${pre}_hs${HIDDEN}_mb${MICRO_BATCH}"
    pre="${pre}_seq${SEQ}_gb${GLOBAL_BATCH}"
    pre="${pre}_sp${SP}_pp${PP}_tp${TP}_${DTYPE}_opt${OPT}"
    pre="${pre}_lr${LR}_lwf${LR_WARMUP_FRAC}"
    if [[ -n "${TOKENIZER_TYPE:-}" ]]; then
        pre="${pre}_tok${TOKENIZER_TYPE}"
    fi
    if [[ -n "${LR_DECAY_ITERS}" ]]; then
        pre="${pre}_ldi${LR_DECAY_ITERS}"
    fi
    if [[ -z "${NO_FLASH_ATTN:-}" ]]; then
        pre="${pre}_flash"
    fi
    export OUTPUT_PREFIX="${pre}"
    echo "${pre}"
}

setOutput() {
    # OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
    OUTPUT_PREFIX=$(get_output_prefix)
    OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%Y%m%d-%H%M%S)_${WORLD_SIZE}_${HOSTNAME}"
    export OUTPUT_DIR="${OUTPUT_DIR}" && mkdir -p "${OUTPUT_DIR}"
    export OUTPUT_LOG="${OUTPUT_DIR}/output.log"
    export CKPT_DIR="checkpoints/${OUTPUT_PREFIX}"
    echo "${OUTPUT_LOG}" >>"logs/latest"
    printf "\n Please see logs at: %s\n" "$(printGreen "${OUTPUT_DIR}")"
    printf "Checkpoints will be saved to: %s\n" "$(printYellow "${CKPT_DIR}")"
}

#############################################
# Build DeepSpeed config and write to .json
#############################################
buildDSconfig() {
    export CPU_OPTIMIZER="${CPU_OPTIMIZER:-0}"
    export DS_CONFIG="${WORKING_DIR}/ds-configs/ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
    mkdir -p "$(dirname "${DS_CONFIG}")"
    echo "DS_CONFIG: ${DS_CONFIG}"
    printf "ZS: %s, MB: %s, GB: %s, PP: %s, DTYPE: %s" "${ZERO_STAGE}" "${MICRO_BATCH}" "${GLOBAL_BATCH}" "${PP}" "${DTYPE}"
    generateDSconfig "${DS_CONFIG}"
}

###############################################################################
# sumWeights
#
# This will sum the weights (first column) from each line in the passed
# `file_list`.
###############################################################################
sumWeights() {
    local file_list=$1
    weights=$(cat "${file_list}" | awk '{print $1}' | tr '\n' '\ ,\ ' | sed 's/^/[/g' | sed 's/$/]/g' | tr '\ ' "\,\ ")
    python3 -c "import numpy as np; print(np.sum(${weights}))"
}

sumFiles() {
    local rd=$1
    for f in $("${rd}/*.txt"); do
        ws=$(sumWeights "${rd}/${f}")
        echo "sum($f.weights)=${ws}"
    done
}

###########################################
# make_data
#
# This will run `make` in `megatron/data`
# prior to launching, ensuring that
# `megatron/data/helpers.cpp`
# is built appropriately.
###########################################
make_data() {
    python3 -m pip install pybind11
    mdir="${WORKING_DIR}/megatron/data"
    cd "${mdir}" && make && cd -
}

##############################################################################
# install_dependencies
#
# Ensure all dependencies installed from `ALCF/requirements/requirements.txt`
##############################################################################
install_dependencies() {
    depsfile="${WORKING_DIR}/ALCF/requirements/requirements.txt"
    echo "[install_dependencies] Ensuring all dependencies from ${depsfile} installed..."
    python3 -m pip install -r "${depsfile}" --require-virtualenv 1>/dev/null
    if [[ ! -x "$(command -v deepspeed)" ]]; then
        mn=$(get_machine_name)
        # if [[ "${mn}" == aurora* || "${mn}" == sunspot* ]]; then
        #     install_deepspeed_for_xpu || exit
        # fi
        printf "[install_dependencies] No 'deepspeed' command found on %s" "${mn}"
        printf "[install_dependencies] !! No deepsepeed in %s" "$(which python3)"
    fi
}

######################################################################
# install_deepspeed_for_xpu
#
# Install microsoft/DeepSpeed on PVC
#
# This will:
# 1. Clone rep
# 2. Checkout appropriate branch
# 3. Install into virtual environment
######################################################################
install_deepspeed_for_xpu() {
    # python3 -m pip install "torch==2.1.0.post2" torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 oneccl_bind_pt==2.1.300+xpu --extra-index-url "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    echo "Building + Installing DeepSpeed on $(hostname)"
    outdir="${WORKING_DIR}/deps/DeepSpeed"
    mkdir -p "${outdir}"
    git clone https://github.com/microsoft/DeepSpeed.git "${outdir}"
    cd "${outdir}" || exit
    echo "[install_deepspeed_for_xpu] !! pwd: $(pwd)"
    python3 -m pip install --require-virtualenv -r requirements/requirements.txt 1>/dev/null
    python3 -m pip install xgboost "numpy<2" --force-reinstall --upgrade --require-virtualenv 1>/dev/null
    python setup.py develop 1>/dev/null
    cd "${WORKING_DIR}"
    echo "[install_deepspeed_for_xpu] !! pwd: $(pwd)"
}

#################################################
# Fix for distributed key value store on Aurora
#################################################
use_kvs_fix_on_aurora() {
    export CCL_KVS_MODE=mpi
    export CCL_CONFIGURATION_PATH=""
    export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LD_LIBRARY_PATH
    export CPATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/include:$CPATH
    export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LIBRARY_PATH
    #########################################################
    # if not set, CCL will complain... ?
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
    #########################################################
}

update_ccl_env_vars_aurora() {
    # export CCL_KVS_MODE=mpi
    # # export CCL_CONFIGURATION_PATH=""
    # # unset CCL_CONFIGURATION_PATH
    # # export CCL_CONFIGURATION=cpu_gpu_dpcpp
    # # export CCL_ROOT="/flare/Aurora_deployment/intel/ccl/_install_release_2021_13"
    # export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LD_LIBRARY_PATH
    # export CPATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/include:$CPATH
    # export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LIBRARY_PATH
    # # export CCL_ALLREDUCE_SCALEOUT=direct
    # printenv | grep -E -v "^__" | grep -E "CCL|LD|CPATH|LIBRARY_PATH"
    #########################################################
    # if not set, CCL will complain... ?
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
    #########################################################
    # Sam: [2024-06-29]
    export CCL_KVS_MODE=mpi
    export CCL_CONFIGURATION_PATH=""
    export CCL_CONFIGURATION=cpu_gpu_dpcpp
    export CCL_ROOT="/flare/Aurora_deployment/intel/ccl/_install_release_2021_13"
    export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LD_LIBRARY_PATH
    export CPATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/include:$CPATH
    export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LIBRARY_PATH
}

##########################################################
# Check that we can find the `.py` file we wish to launch
##########################################################
check_executable() {
    fp=$1
    if [[ -f "${fp}" ]]; then
        export EXEC="${fp}"
        # ----[1.5 Keep track of stem from file path]-------------------------
        exec_stem=$(echo "${EXEC}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.py//g")
        export EXEC_STEM="${exec_stem}"
    else
        estr="Unable to locate executable ${fp}"
        printf "[ALCF.helpers:check_executable] %s\n" "$(printRed "${estr}")"
    fi
}

######################################################################
# `makeHostiles`:
#     Detect if `HOSTFILE` set in active environment.
#         - If so, use this.
#         - Otherwise, make default HOSTFILEs from "${PBS_NODEFILE}"
######################################################################
makeHostfiles() {
    if [[ -n "${HOSTFILE}" ]]; then
        printf "!! USING CUSTOM HOSTFILE FROM: %s" "${HOSTFILE}"
    else
        make_ds_hostfile
    fi
}

##################################################
# Setup tokenizer as either Llama2 or GPT2 style
##################################################
setup_tokenizer_and_data() {
    if [[ "$#" == 1 ]]; then
        tok="$1"
        dfl="${DATA_FILE_LIST:-}"
    elif [[ "$#" == 2 ]]; then
        tok="$1"
        dfl="$2"
    else
        echo "Incorrect number of arguments passed. Received: $#, expected 2"
    fi
    echo "Setting up tokenizer with ${tok}"
    echo "Using data_file_list: ${dfl}"
    if [[ ${tok} == gpt* || ${tok} == GPT* ]]; then
        export TOKENIZER_TYPE="GPT2"
        export TOKENIZER_FLAGS="--tokenizer-type GPT2BPETokenizer"
        machine=$(get_machine_name)
        if [[ ${machine} == "polaris" ]]; then
            export DATA_PARENT="${DATA_PARENT:-/eagle/argonne_tpc/foremans/projects/argonne-lcf/Megatron-DeepSpeed/dataset}"
        elif [[ ${machine} == "sunspot" ]]; then
            export DATA_PARENT="${DATA_PARENT:-/gila/Aurora_deployment/foremans/anl_24_q2_release/Megatron-DeepSpeed/dataset}"
        elif [[ ${machine} == "aurora" ]]; then
            export DATA_PARENT="${DATA_PARENT:-/gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/dataset}"
        else
            export DATA_PARENT="${DATA_PARENT:-${WORKING_DIR}/dataset}"
        fi
        export VOCAB_FILE="${DATA_PARENT}/gpt2-vocab.json"
        export MERGE_FILE="${DATA_PARENT}/gpt2-merges.txt"
        export DATA_PATH="${DATA_PARENT}/BookCorpusDataset_text_document"
        export DATA_FLAGS="--data-path ${DATA_PATH} --vocab-file ${VOCAB_FILE} --merge-file ${MERGE_FILE}"
    else
        export DATA_FLAGS=""
        export TOKENIZER_TYPE="Llama2"
        tm="${WORKING_DIR}/ALCF/tokenizer.model"           # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
        export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${tm}}" # USE TOKENIZER_MODEL from env, else fallback from ^
        export TOKENIZER_FLAGS="--tokenizer-type Llama2Tokenizer --tokenizer-model ${TOKENIZER_MODEL}"
        if [[ "${TOKENIZER_TYPE}" != "GPT2" ]]; then
            echo "Using tokenizer: ${TOKENIZER_TYPE}. Setting up data with ${DATA_FILE_LIST-}"
            setData "${dfl}" || exit
        fi
    fi
    printf "[setData] DATA_FLAGS: %s\n" "$(printGreen "${DATA_FLAGS}")"
    printf "[setData] TOKENIZER_FLAGS: %s\n" "$(printMagenta "${TOKENIZER_FLAGS}")"
}

###############################################
# setData
#
# Ensure `DATA_FILE_LIST` is set,
# fallback to default values if necessary.
###############################################
setData() { # ------------------------[dfl: abbrv. for DATA_FILE_LIST]
    ####### [Set DATA_FILE_LIST_FALLBACK based on current machine] #############
    mn=$(get_machine_name)
    dfl_fallback="${WORKING_DIR}/ALCF/data-lists/${mn}/dolma.txt"
    ############################################################################
    # set `dfl` to `dfl_fallback` if not passed as an argument,
    # use this data file list to call `setData`
    dfl="${1:-${dfl_fallback}}"
    printf "Calling:  setData() with %s\n" "${dfl}"
    ndocs=$(wc -l <"${dfl}")
    ws=$(sumWeights "${dfl}")
    dfl_stem=$(echo "${dfl}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
    dcp=".cache/${dfl_stem}/index-cache"
    export DATA_FILE_LIST="${dfl}"
    export NUM_DOCS="${ndocs}"
    export WEIGHT_SUM="${ws}"
    export DFL_STEM="${dfl_stem}"
    export DATA_CACHE_PATH="${dcp}"
    export DATA_FLAGS="${DATA_FLAGS} --data-file-list ${DATA_FILE_LIST}" #  --data-cache-path ${DATA_CACHE_PATH}"
    echo "--------------------"
    echo "Updated environment:"
    printf "DATA_FILE_LIST: %s\n" "${DATA_FILE_LIST}"
    printf "NUM_DOCS: %s\n " "${NUM_DOCS}"
    printf "WEIGHT_SUM: %s\n" "${WEIGHT_SUM}"
    printf "DFL_STEM: %s\n" "${DFL_STEM}"
    printf "DATA_CACHE_PATH: %s\n" "${DATA_CACHE_PATH}"
    printf "DATA_FLAGS: %s\n" "${DATA_FLAGS}"
    echo "--------------------"
}

################################################################################
# generateDSconfig
#
# Create and save a deepspeed config .json
#
# This will contain the appropriate variables as set in the current environment.
################################################################################
generateDSconfig() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 config_file"
        exit 1
    fi
    for v in "$GLOBAL_BATCH" "$MICRO_BATCH" "$GRAD_ACC_STEPS" "$ZERO_STAGE" "$PP" "$DTYPE"; do
        if [ -z "$v" ]; then
            echo "Please export required envs before execute $0"
            exit 1
        fi
    done
    # \"optimizer\": {
    #   \"type\": \"AdamW\",
    #   \"params\": {
    #     \"lr\": ${LR},
    #     \"beta1\": 0.9,
    #     \"beta2\": 0.95,
    #     \"eps\": 1e-5,
    #     \"weight_decay\": 1e-1
    #   }
    # },
    # \"scheduler\": {
    #   \"type\": \"WarmupLR\",
    #   \"params\": {
    #       \"warmup_min_lr\": 0.00003,
    #       \"warmup_max_lr\": 0.0003,
    #       \"warmup_num_steps\": 5000
    #   }
    # },
    extra=""
    common="\
        \"train_batch_size\": $GLOBAL_BATCH,
        \"train_micro_batch_size_per_gpu\": $MICRO_BATCH,
        \"steps_per_print\": 1,
        \"gradient_accumulation_steps\": $GRAD_ACC_STEPS,
        \"zero_allow_untested_optimizer\": true,
        \"gradient_clipping\": 1.0,
        \"activation_checkpointing\": {
          \"partition_activations\": true,
          \"contiguous_memory_optimization\": true
        },
        \"wall_clock_breakdown\": false,"
    flops_profiler="\
        \"flops_profiler\": {
          \"enabled\": true,
          \"profile_step\": 2,
          \"module_depth\": -1,
          \"top_modules\": 1,
          \"detailed\": true,
          \"output_file\": null
        }"
    if [[ $DTYPE == "bf16" ]]; then
        dtype="\
            \"communication_data_type\": \"bf16\",
            \"fp16\": {
              \"enabled\": false,
              \"loss_scale\": 0,
              \"loss_scale_window\": 1000,
              \"hysteresis\": 2,
              \"min_loss_scale\": 1
            },
            \"bfloat16\": {
              \"enabled\": true,
              \"loss_scale\": 1.0
            },"
    elif [[ $DTYPE == "fp16" ]]; then
        dtype="\
            \"communication_data_type\": \"fp16\",
            \"fp16\": {
              \"enabled\": true,
              \"loss_scale\": 0,
              \"loss_scale_window\": 1000,
              \"hysteresis\": 2,
              \"min_loss_scale\": 1
            },
            \"bfloat16\": {
              \"enabled\": false,
              \"loss_scale\": 1.0
            },"
    else
        dtype="\"communication_data_type\": \"fp32\","
    fi
    if [[ "${ZERO_STAGE}" == 3 ]]; then
        # \"mics_shard_size\": 2,
        zero="\
            \"zero_optimization\": {
              \"stage\": 3,
              \"reduce_scatter\": false,
              \"mics_hierarchical_params_gather\": true,
              \"stage3_max_live_parameters\": 3e9,
              \"stage3_max_reuse_distance\": 3e9,
              \"stage3_param_persistence_threshold\": 1e5,
              \"stage3_prefetch_bucket_size\": 5e7,
              \"contiguous_gradients\": true,
              \"overlap_comm\": true,
              \"reduce_bucket_size\": 90000000,
              \"sub_group_size\": 1e9,
              \"offload_optimizer\": {
                \"device\": \"none\",
                \"buffer_count\": 4,
                \"pipeline_read\": false,
                \"pipeline_write\": false,
                \"pin_memory\": true
              }
            },"
    # elif [[ $ZERO_STAGE == 2 ]]; then
    elif [[ "${ZERO_STAGE}" == 2 || "${ZERO_STAGE}" == 1 ]]; then
        # if [[ -n "${CPU_OPTIMIZER}" ]]; then
        if [[ "${CPU_OPTIMIZER:-0}" != 0 ]]; then
            echo "!!!! CAUGHT CPU_OPTIMIZER !!!!"
            zero="\
                \"zero_optimization\": {
                    \"stage\": $ZERO_STAGE,
                    \"offload_optimizer\": {
                      \"device\": \"cpu\"
                    }
                },"
        else
            zero="\
                \"zero_optimization\": {
                  \"stage\": $ZERO_STAGE
                },"
        fi
        if [[ "${PP}" -gt 1 ]]; then
            extra="\
                \"data_types\": {
                \"grad_accum_dtype\": \"fp32\"
              },
              \"comms_logger\": {
                \"enabled\": true,
                \"verbose\": false,
                \"prof_all\": true,
                \"debug\": false
              },"
        else
            extra="\
                \"comms_logger\": {
                \"enabled\": true,
                \"verbose\": false,
                \"prof_all\": true,
                \"debug\": false
              },"
        fi
    else
        echo 'Please add the correct config set!!!'
    fi
    cat <<EOT >"$1"
{
$common
$zero
$dtype
$extra
$flops_profiler
}
EOT
}

# #####################
# # train
# #####################
# train() {
#     # 1. Navigate into `$PBS_O_WORKDIR` <-- [should be Megatron-Deepspeed]
#     cd "${PBS_O_WORKDIR}" || exit
#     HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
#     # 2. source `ALCF/helpers.sh` <-- [should be ./ALCF/helpers.sh]
#     source "${HERE}/ALCF/helpers.sh" || exit
#     # 3. call `setup` from `./ALCF/helpers.sh`
#     # export DATA_FILE_LIST="${HERE}/ALCF/data-lists/$(get_machine_name)/books.txt"
#     setup || exit
#     # 4. Take custom args
#     export custom_args=" $@"
#     # 5. Update ${run_cmd} (from setup ALCF/helpers.sh) with ${custom_args}
#     export run_cmd="${run_cmd} ${custom_args}"
#     # 6. Add "${run_cmd}" to output log
#     echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
#     # 7. Tell user where to find output
#     printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
#     # 8. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
#     eval "${run_cmd}" |& tee -a "${OUTPUT_LOG}"
#     set +x
# }

###############################################
# Helper functions for printing colored text
###############################################
RESET="\e[0m"
BLACK="\e[1;30m"
RED="\e[1;31m"
GREEN="\e[1;32m"
YELLOW="\e[1;33m"
BLUE="\e[1;34m"
CYAN="\e[1;35m"
# WHITE="\e[1;36m"

printBlack() {
    printf "\e[1;30m%s\e[0m\n" "$@"
}

printRed() {
    printf "\e[1;31m%s\e[0m\n" "$@"
}

printGreen() {
    printf "\e[1;32m%s\e[0m\n" "$@"
}

printYellow() {
    printf "\e[1;33m%s\e[0m\n" "$@"
}

printBlue() {
    printf "\e[1;34m%s\e[0m\n" "$@"
}

printMagenta() {
    printf "\e[1;35m%s\e[0m\n" "$@"
}

printCyan() {
    printf "\e[1;36m%s\e[0m\n" "$@"
}

printWhite() {
    printf "\e[1;37m%s\e[0m\n" "$@"
}

###########################
# call helpers_main()
###########################
helpers_main

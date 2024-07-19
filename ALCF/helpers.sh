#!/bin/bash --login

###############################################################################
# `ALCF/helpers.sh`
# <https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh>
#
# Contains helper functions for launching `../train_llama_alcf.sh`
#
# > [!NOTE]
# > On any of {Polaris, Aurora, Sunspot} @ ALCF:
# >
# > ```bash
# > $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
# > $ cd Megatron-DeepSpeed
# > $ PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh
# > # on any of {Polaris, Sunspot, Aurora} @ ALCF
# > $ setup_python
# > ```
##############################################################################

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
#       1. `${PBS_O_WORKDIR}` is nonzero, use this
#       2. else, if `${SLURM_SUBMIT_DIR}` is nonzero use this
#       3. else, use `$(pwd)`
#
#   this is crucial since many of the functions below use paths 
#   which are defined relative to this "${WORKING_DIR}"
#   (e.g. virtual environment, location of executables, etc.)
##################
helpers_main() {
    # for debug mode, run with `DEBUG=1`
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
#      1. Identify the machine we're on
#      2. Setup `python`
#         1. Load `conda`
#         2. Setup `venv` on top of `conda`
#      3. Ensure all dependencies are installed
#      4. Clone + Install [`saforem2/ezpz`](https://github.com/saforem2/ezpz)
#         1. Additionally, call `source deps/ezpz/src/ezpz/bin/savejobenv`,
#            which will automatically build a `alias launch=mpiexec ...`
#            according to the specifics of our active job.
#      5. Set runtime options
#      6. Build `deepspeed_config.json`
#      7. Build {logs, checkpoints, etc} dirs, named according to specifics of
#         current run
#      8. Specify additional `deepspeed` arguments
#      9. Ensure executable exists at expected path
#     10. Setup data + tokenizer via `TOKENIZER_TYPE`
#     11. Print job info
#     12. Save `.env` to `CKPT_DIR` for safe keeping
#     13. Check that we're not already running, and if so, exit.
#     14. Setup run command to be executed.
##############################################################################
setup() {
    #  1. Identify machine we're on
    get_machine || exit
    #  2. Load `conda` environment, setup virtual env
    setup_python || exit
    #  3. Ensure necessary dependencies all installed
    install_dependencies || exit
    #  4. Determine WORLD_SIZE, etc. from `PBS_*` vars
    setup_ezpz || exit
    #  5. Set command line arguments to pass to `"${EXEC}"`
    setParams || exit
    #  6. Create `deepspeed_config.json` from runtime params from ^
    buildDSconfig || exit
    #  7. Specify output directory for {logs, checkpoints, etc.}
    setOutput || exit
    #  8. Specify additional `deepspeed` arguments (dependent on _newly created_ variables)
    setArgs || exit
    #  9. Ensure executable exists in expected path
    export EXEC="${EXEC:-${HERE}/pretrain_gpt_alcf.py}"
    check_executable "${EXEC}"
    dfl="${DATA_FILE_LIST:-}"
    # 10. Setup data + tokenizer via `DATA_FILE_LIST` and `TOKENIZER_TYPE`
    tok="${TOKENIZER_TYPE:-Llama2}"
    setup_tokenizer_and_data "${tok}" "${dfl}" || exit
    make_data || exit
    # 11. Print job info
    printJobInfo || exit
    # 12. Save `.env` to `CKPT_DIR` for safe keeping
    save_dotenv "${CKPT_DIR}" || exit
    # 13. Check that were not already running, if so, exit.
    check_and_kill_if_running || exit
    # 14. Setup run command to be executed
    setup_run_cmd || exit
}

#####################################################
# setup_run_cmd
#
# Build run command to be executed.
#####################################################
setup_run_cmd() {
    #### Make it easy to track experiments by date ###################
    year="$(date "+%Y")"
    month="$(date "+%m")"
    day="$(date "+%Y-%m-%d")"
    today="$(date "+%Y-%m-%d")"  # kept for backwards compatibility
    started_at="$(date "+%Y-%m-%d-%H%M%S")"
    export YEAR="${year}"
    export MONTH="${month}"
    export DAY="${day}"
    export TODAY="${today}"
    export STARTED_AT="${started_at}"
    ##################################################################
    # to launch with DeepSpeed instead of mpiexec, run with:
    #       $ LAUNCH_WITH=deepspeed bash train_llama_alcf.sh
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
        --use-checkpoint-opt_param-scheduler \
        --log-timers-to-tensorboard \
        --log-optimizer-states-to-tensorboard"
    if [[ "${SP}" -ge 2 ]]; then
        export DEFAULTS="${DEFAULTS} --ds-sequence-parallel-size ${SP} --force-ds-sequence-parallel"
    fi
    # [hacky]: to disable Llama-type architectures, toggle via:
    # `NO_LLAMA=1 bash train_llama_alcf.sh`
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
        --save ${CKPT_DIR} \
        --load ${CKPT_DIR} \
        --seq-length ${SEQ} \
        --num-layers ${NLAYERS} \
        --hidden-size ${HIDDEN} \
        --train-iters ${TRAIN_ITER} \
        --tensorboard-dir ${TBDIR} \
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
        ${gpt_args[*]}
        "
        # ${LLAMA_ARGS} \
}

save_dotenv() {
    if [[ "$#" -ne 1 ]]; then
        estr="[error]"
        # echo "Expected exactly one argument, specifying outputdir. Received $#"
        printf "%s Expected one argument (outdir). Received: %s" "$(printRed "${estr}")" "$#"
    else
        outdir="$1"
        mkdir -p "${outdir}"
        module list
        dotenv_file="${outdir}/.env"
        echo "Saving environment to ${dotenv_file}"
        printenv | grep -v "LS_COLORS" > "${dotenv_file}"
        export DOTENV_FILE="${dotenv_file}"
    fi
}

######################################################################
# get_machine_name: Return current machine name, as lowercase string
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
    # kill $(ps aux | grep -E "$USER.+(mpi|main.py)" | grep -v grep | awk '{print $2}')
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]];
        then echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}";
    else
        echo "Not currently running. Continuing!"
    fi
}


setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
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
        if [[ -n "${DIST_LAUNCH}" ]]; then
            LAUNCHER="${DIST_LAUNCH} --genvall $(which python3) -Wignore ${EXEC}"
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
        nhosts=$(wc -l < "${HOSTFILE:-${PBS_NODEFILE}}")
        if [[ "${nhosts}" == 1  || "${nhosts}" == 2 ]]; then
            mbs=1
        elif [[ "${nhosts}" -ge 3 && "${nhosts}" -le 7 ]]; then
            mbs=2
        elif [[ "${nhosts}" -ge 8 ]]; then
            mbs=4
        fi
    fi
    echo "${mbs}"
}


##############################################################################
# `setParams`: Set / configure run options by parsing environment.
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
    # +----[Parallelism Settings] -------------------------------------------+
    # +------[Aurora]--------||-------[SunSpot]-------------+
    if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        TP=${TP:-1}                      # TP = 1
        export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
        export CCL=${CCL:-ccl}           # CCL
        export BE="${CCL}"               # COMMUNICATION BACKEND = CCL
        export DTYPE=${DTYPE:-bf16}      # DTYPE: bf16
        export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}     # GRADIENT_ACC_STEPS
        MICRO_BATCH=${MICRO_BATCH:-4}    # MICRO_BATCH = 4
        ######################################################################
        # !XXX: USE KEY VALUE STORE FIX ON AURORA [2024-06-20]
        # use_kvs_fix_on_aurora  # <-- why are these different from those in update_ccl_env_vars_aurora ??
        update_ccl_env_vars_aurora
        ######################################################################
        if [[ -z "${USE_FLASH_ATTN:-}" ]]; then
            # NOTE: if NO_FLASH_ATTN is NON-empty; then NO FLASH ATTN !!
            export NO_FLASH_ATTN=1 # disabled on [2024-06-20] waiting on fix...
            if [[ -n "${NO_FLASH_ATTN-}" ]]; then
                echo "Not using flash-attn!!"
            else
                # LLAMA_ARGS="${LLAMA_ARGS} --use-flash-attn-builder"
                FLASH_ARG="--use-flash-attn-builder"
            fi
        else
            echo "Using flash-attn !!"
            FLASH_ARG="--use-flash-attn-builder"
        fi
        ######################################################################
    # +--------[Polaris]-----------------------------------+
    elif [[ $(hostname) == x3* ]]; then
        # export LAUNCH_CMD="${LAUNCH_CMD:-deepspeed}"
        TP=${TP:-1}                                     # TP = 2
        export NCCL=${NCCL:-nccl}                       # NCCL
        export BE="${NCCL}"                             # BE = NCCL
        # export DTYPE=${DTYPE:-bf16}                   # DTYPE: BF16 ??
        export DTYPE=${DTYPE:-fp16}                     # DTYPE: FP16
        export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}      # GRADIENT_ACC_STEPS
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
    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
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
    export TP="${TP}"
    export PP="${PP:-1}"
    export SP="${SP:-1}"
    export FLASH_ARG="${FLASH_ARG}"
    export DTYPE="${DTYPE:-bf16}"
    export OPT="${OPT:-adamw}"
    export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
    NHOSTS=$(wc -l < "${HOSTFILE}")
    if [[ -z "${NGPU_PER_HOST:-}" ]]; then
        NGPU_PER_HOST=$(python3 -c 'import ezpz as ez; print(ez.get_gpus_per_node())')
    fi
    export NGPU_PER_HOST="${NGPU_PER_HOST}"
    export WORLD_SIZE="${WORLD_SIZE:-$(( NHOSTS * NGPU_PER_HOST ))}"
    # +---[Llama2 7B Config]--------------------------------------------------+
    # export MODEL_KEY="Llama-7B"
    export HEADS=${HEADS:-${NHEADS:-32}}                # NUMBER OF ATEN HEADS
    export NLAYERS=${NLAYERS:-${NUM_LAYERS:-32}}        # NUMBER OF LAYERS
    export HIDDEN=${HIDDEN:-4096}                       # HIDDEN SIZE
    export NUM_KV_HEAD=${NUM_KV_HEAD:-8}                # GROUP ATTENTION
    export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}    # FFN HIDDEN SIZE
    # +---[Run Settings]------------------------------------------------------+
    export SEQ=${SEQ:-4096}                             # SEQ_LEN: 4096
    export ZERO_STAGE=${ZERO_STAGE:-1}                  # ZERO OFFLOADING STAGE
    export MICRO_BATCH=${MICRO_BATCH:-8}                # MICRO BATCH SIZE
    export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}          # GRADIENT ACCUMULATION STEPS
    export EVAL_ITERS="${EVAL_ITERS:-10}"               # NUMBER OF EVAL ITERS TO RUN
    export TRAIN_ITER=${TRAIN_ITER:-317892}             # NUMBER OF TRAIN ITERS
    export EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"      # HOW FREQUENTLY TO RUN EVAL
    export SAVE_INTERVAL=${SAVE_INTERVAL:-50}           # HOW FREQUENTLY TO SAVE CKPTS
    export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"    # TIMING VERBOSITY IN LOGS
    export ACT_CKPT_NUM_LAYERS="${ACT_CKPT_NUM_LAYERS:-1}"                  # NUM LAYERS TO CHECKPOINT ACTIVATIONS
    export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}  # USE ACTIVATION CHECKPOINTING ?
    export GLOBAL_BATCH_MAX=$(( WORLD_SIZE * MICRO_BATCH * GRAD_ACC_STEPS / TP / PP  / SP ))  # MAX GLOBAL BATCH SIZE
    export GLOBAL_BATCH="${GLOBAL_BATCH:-${GLOBAL_BATCH_MAX}}"  # WILL USE MAX IF NOT SET IN ENVIRONMENT
    # tm="${WORKING_DIR}/ALCF/tokenizer.model"            # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
    # export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${tm}}"  # USE TOKENIZER_MODEL from env, else fallback from ^
    export MODEL_TYPE="llama-gb${GLOBAL_BATCH}-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"  # STRING FOR IDENTIFYING MODEL
    # +----[ADDITIONAL LLAMA SPECIFIC ARGUMENTS]------------------------------
    if [[ "${SP}" == 1 ]]; then
        export LLAMA_ARGS="${LLAMA_ARGS} --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
    else
        export LLAMA_ARGS=""
        echo "NOT USING ROTARY EMBEDDINGS! LLAMA_ARGS=${LLAMA_ARGS}"
    fi
    # -----[Learning Rate Settings]--------------------------------------------
    export LR=${LR:-0.0003}                             # LEARNING_RATE
    export LR_WARMUP_FRAC=${LR_WARMUP_FRAC:-0.05}       # LEARNING RATE WARMUP
    # export LR_DECAY_ITERS=${LR_DECAY_ITERS:-320000}     # LR DECAY ITERS
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
# setArgs
#
# Specify additional (DeepSpeed specific)
# arguments to pass to pretrain_gpt_alcf.py
##############################################
setArgs() {
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
    cat "${hf}" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "${hf}" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}

####################
# ezpz_savejobenv
#
# Parse relevant environment variables to determine:
# - num_hosts (by counting the number of lines in ${HOSTFILE:-${PBS_NODEFILE}})
# - num_gpus_per_host (magically[^1])
# - num_gpus = (num_hosts * num_gpus_per_host)
#
# [^1]: See: [`ezpz/bin/savejobenv`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/savejobenv)
####################
ezpz_savejobenv() {
    # source "${WORKING_DIR}/deps/ezpz/src/ezpz/bin/savejobenv" "$@"
    file=$(mktemp)
    curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/savejobenv > "${file}"
    source "${file}" || exit
}

ezpz_getjobenv() {
    # source "${WORKING_DIR}/deps/ezpz/src/ezpz/bin/getjobenv" "$@"
    file=$(mktemp)
    curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/getjobenv > "${file}"
    source "${file}" || exit
}

###########################################
# setup_ezpz
#
# 1. {save,get}jobenv
# 2. python3 -m ezpz.jobs && source "./.jobenv"
# 2. Install ezpz (if not installed)
###########################################
setup_ezpz() {
    if [[ -n "${HOSTFILE:-${PBS_NODEFILE}}" ]]; then
        ezpz_savejobenv
    else
        ezpz_getjobenv
    fi
    ezdir="${WORKING_DIR}/deps/ezpz"
    if [[ ! -d "${ezdir}" ]]; then
        mkdir -p "${WORKING_DIR}/deps"
        git clone https://github.com/saforem2/ezpz "${WORKING_DIR}/deps/ezpz"
    fi
    ezloc=$(python3 -m pip list | grep ezpz | awk '{print $NF}')
    if [[ -z "${ezloc:-}" ]]; then
        printf "[setup_ezpz] Installing ezpz from %s\n" "${ezdir}"
        python3 -m pip install -e "${ezdir}" --require-virtualenv
    else
        printf "[setup_ezpz] Found ezpz @ %s\n" "${ezloc}"
    fi
    python3 -m ezpz.jobs && source "./.jobenv"
    # ezloc=$(python3 -m pip list | grep ezpz | awk '{print $NF}')
    # if [[ -n "${ezloc}" ]]; then
    #     # ezpz_savejobenv
    #     # python3 -m ezpz.jobs && source "./.jobenv"
    #     make_ds_hostfile || exit
    # else
    #     echo "No ezpz detected. Attempting to install with $(which python3)"
    #     python3 -m pip install -e "${WORKING_DIR}/deps/ezpz" --require-virtualenv
    # fi
    # if [[ ! -d "${WORKING_DIR}/deps/ezpz" ]]; then
    #     mkdir -p "${WORKING_DIR}/deps"
    #     git clone https://github.com/saforem2/ezpz "${WORKING_DIR}/deps/ezpz"
    # else
    #     echo "Found ezpz!"
    # fi
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
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy:-}" ;
        echo "https_proxy=${https_proxy:-}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
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
    echo "${OUTPUT_LOG}" >> "logs/latest"
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


install_dependencies() {
    depsfile="${WORKING_DIR}/ALCF/requirements/requirements.txt"
    echo "Ensuring all dependencies from ${depsfile} installed..."
    python3 -m pip install -r "${depsfile}" --require-virtualenv 1> /dev/null
    if [[ ! -x "$(command -v deepspeed)" ]]; then
        mn=$(get_machine_name)
        if [[ "${mn}" == aurora* || "${mn}" == sunspot* ]]; then
            install_deepspeed_for_xpu || exit
        fi
        printf "!! No 'deepspeed' command found on %s" "${mn}"
        printf "!! No deepsepeed in $(which python3)"
    fi
}

######################################################################
# install_deepspeed_for_xpu: Install microsoft/DeepSpeed on PVC
#
# This will:
# 1. Clone rep
# 2. Checkout appropriate branch
# 3. Install into virtual environment
######################################################################
install_deepspeed_for_xpu() {
    echo "Building + Installing DeepSpeed on $(hostname)"
    outdir="${WORKING_DIR}/deps/DeepSpeed"
    mkdir -p "${outdir}"
    git clone https://github.com/microsoft/DeepSpeed.git "${outdir}"
    cd "${outdir}" || exit
    echo "!! pwd: $(pwd)"
    git remote add yizhou_ds https://github.com/YizhouZ/DeepSpeed.git
    git fetch yizhou_ds
    git checkout yizhou/kernel_path
    python3 -m pip install --require-virtualenv -r requirements/requirements.txt 1> /dev/null
    python3 -m pip install xgboost "numpy<2" --force-reinstall --upgrade --require-virtualenv 1> /dev/null
    python setup.py develop 1> /dev/null
    cd "${WORKING_DIR}"
    echo "!! pwd: $(pwd)"
}


########################################################
# Setup / activate conda environment(s)
########################################################

###########################
# Setup conda on Sunspot
###########################
setup_conda_sunspot() {
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /soft/preview-modulefiles/24.086.0
        module load frameworks/2024.04.15.002.lua
        # module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
        # source "${WORKING_DIR}/ALCF/sunspot-env-2024-q2.sh"
    fi
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
    export CCL_KVS_MODE=mpi
    # export CCL_CONFIGURATION_PATH=""
    # unset CCL_CONFIGURATION_PATH
    # export CCL_CONFIGURATION=cpu_gpu_dpcpp
    # export CCL_ROOT="/flare/Aurora_deployment/intel/ccl/_install_release_2021_13"
    export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LD_LIBRARY_PATH
    export CPATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/include:$CPATH
    export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LIBRARY_PATH
    # export CCL_ALLREDUCE_SCALEOUT=direct
    printenv | grep -E -v "^__" | grep -E "CCL|LD|CPATH|LIBRARY_PATH"
    #########################################################
    # if not set, CCL will complain... ?
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
    #########################################################
}

###########################
# Setup conda on Aurora
###########################
setup_conda_aurora() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use -a /soft/modulefiles ; module load frameworks/2024.1
    fi
}

########################
# Setup conda on Sirius
########################
setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

########################
# Setup conda on Polaris
########################
setup_conda_polaris() {
    # unset MPICH_GPU_SUPPORT_ENABLED
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX-}" ]]; then
        # if so, load the default conda/2024-04-29
        # module and activate base environment
        module use /soft/modulefiles ; module load conda ; conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

########################
# setup_venv_from_conda
#
# Build (if necessary) a virtual environment
# on top of the active conda and
# activate it.
# ######################
setup_venv_from_conda() {
    if [[ -z "${CONDA_PREFIX}" ]]; then
        echo "!! No ${CONDA_PREFIX} found."  #  Exiting."
        # exit 1
    else
        echo "Found conda at: ${CONDA_PREFIX}"
        CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
        export CONDA_NAME
        if [[ -z "${VIRTUAL_ENV}" ]]; then
            echo "No VIRTUAL_ENV found in environment!"
            echo "    - Trying to setup from ${CONDA_PREFIX}"
            export VENV_DIR="${WORKING_DIR}/venvs/${CONDA_NAME}"
            echo "    - Using VENV_DIR=${VENV_DIR}"
            if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
                printf "\n    - Creating a new virtual env on top of %s in %s\n" "$(printBlue "${CONDA_NAME}")" "$(printGreen "${VENV_DIR}")"
                mkdir -p "${VENV_DIR}"
                python3 -m venv "${VENV_DIR}" --system-site-packages
                source "${VENV_DIR}/bin/activate" || exit
            elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
                echo "    - Found existing venv, activating from $(printBlue "${VENV_DIR}")"
                source "${VENV_DIR}/bin/activate"
            else
                printf "\n    [!! %s]: Unable to locate %s\n" "$(printRed "ERROR")" "$(printMagenta "${VENV_DIR}/bin/activate")"
            fi
        fi
    fi

}

##########################################################
# Check that we can find the `.py` file we wish to launch
##########################################################
check_executable() {
    fp=$1
    if [[ -f "${fp}" ]]; then
        export EXEC="${EXEC}"
        # ----[1.5 Keep track of stem from file path]-------------------------
        exec_stem=$(echo "${EXEC}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.py//g")
        export EXEC_STEM="${exec_stem}"
    else
        estr="Unable to locate executable ${fp}"
        printf "[ALCF.helpers:check_executable] %s" "$(printRed "${estr}")"
    fi
}

##############################################################################
# `setup_python`:
#
# 1. Setup `conda`
#    - if `conda` nonempty, and `venv` empty, use `conda` to setup `venv`.
#    - if `venv` nonempty, and `conda` empty, what do (???)
#    - if `venv` nonempty and `conda` nonempty, use these
#    - if `conda` empty and `venv` empty:
#       - if `hostname == x4*`, we're on Aurora
#       - if `hostname == x1*`, we're on Sunspot
#       - if `hostname == x3*`, we're on Polaris
#       - if `hostname == nid*`, we're on Perlmutter
#       - otherwise, you're on you're own
#
# 2. Activate (creating, if necessary) a `venv` on top of `base` conda
#    - use the $CONDA_PREFIX to create a venv in
#      `Megatron-DeepSpeed/venvs/${CONDA_PREFIX}`
#      - activate and use this
#
# 3. Print info about which python we're using
##############################################################################
setup_python() {
    virtual_env="${VIRTUAL_ENV:-}"
    conda_prefix="${CONDA_PREFIX:-}"
    if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix OR virtual_env found in environment..."
        echo "Setting up conda..."
        machine_name=$(get_machine_name)
        echo "machine name: ${machine_name}"
        # if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
        if [[ "${machine_name}" == "aurora" ]]; then
            setup_conda_aurora
        # elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
        elif [[ "${machine_name}" == "sunspot" ]]; then
            setup_conda_sunspot
        # elif [[ $(hostname) == x3*  || $(hostname) == polaris* ]]; then
        elif [[ "${machine_name}" == "polaris" ]]; then
            if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
                setup_conda_sirius
            else
                setup_conda_polaris
            fi
        elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
            echo "Running on Perlmutter !!"
            module load pytorch
            source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
        else # ------------------------------------- [Unknown] -------------------
            echo "Unknown hostname $(hostname)"
            exit 1
        fi
        # # ----- [Perlmutter @ NERSC] -------------------------------------
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: ${conda_prefix}"
        echo "Setting up venv from ${CONDA_PROMPT_MODIFIER:-}"
        setup_venv_from_conda
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "No conda found."
        echo "Using virtual_env from: ${virtual_env}"
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of conda from: ${conda_prefix}"
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    if [[ -z "${virtual_env}" ]]; then
        setup_venv_from_conda
    fi
    pystr="Using: $(which python3)"
    printf "[python] %s" "$(printMagenta "${pystr}")"
    printf "\n"
    export "PYTHON_EXEC=$(which python3)"
}

######################################################################
# `makeHostiles`:
#     Detect if `HOSTFILE` set in active environment.
#         - If so, use this.
#         - Otherwise, make default HOSTFILEs from "${PBS_NODEFILE}"
######################################################################
makeHostfiles() {
    if [[ -n "${HOSTFILE}" ]]; then
        printf "!! USING CUSTOM HOSTFILE FROM: %s"  "${HOSTFILE}"
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
        tm="${WORKING_DIR}/ALCF/tokenizer.model"            # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
        export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${tm}}"  # USE TOKENIZER_MODEL from env, else fallback from ^
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
# `setData`:
#     Ensure `DATA_FILE_LIST` is set,
#     fallback to default values if necessary.
###############################################
setData() {  # ------------------------[dfl: abbrv. for DATA_FILE_LIST]
    ####### [Set DATA_FILE_LIST_FALLBACK based on current machine] #############
    if [[ $(hostname) == x4* ]]; then    # -----------------------------[AURORA]
        dfl_fallback="${WORKING_DIR}/ALCF/data-lists/aurora/dolma.txt"
    elif [[ $(hostname) == x1* ]]; then  # ----------------------------[SUNSPOT]
        # shellcheck source=./data-lists/sunspot/books.txt
        dfl_fallback="${WORKING_DIR}/ALCF/data-lists/sunspot/dolma.txt"
    elif [[ $(hostname) == x3* ]]; then  # -------------------[POLARIS / SIRIUS]
        if [[ "${PBS_O_HOST}" == sirius* ]]; then  # -------------------[SIRIUS]
            # shellcheck source=./data-lists/sirius/books.txt
            dfl_fallback="${WORKING_DIR}/ALCF/data-lists/sirius/dolma.txt"
        elif [[ "${PBS_O_HOST}" == polaris* ]]; then  # ---------------[POLARIS]
            # shellcheck source=./data-lists/polaris/books.txt
            dfl_fallback="${WORKING_DIR}/ALCF/data-lists/polaris/dolma.txt"
        fi
    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then # [PERLMUTTER]
        dfl_fallback="${SLURM_SUBMIT_DIR}/genslm-subsample.txt"
    else  # -----------------------------------------------------------[UNKNOWN]
        echo "Unknown hostname. Must manually specify DATA_FILE_LIST."
    fi
    ############################################################################
    # set `dfl` to `dfl_fallback` if not passed as an argument,
    # use this data file list to call `setData`
    dfl="${1:-${dfl_fallback}}"
    printf "Calling:  setData() with %s\n" "${dfl}"
    ndocs=$(wc -l < "${dfl}")
    ws=$(sumWeights "${dfl}")
    dfl_stem=$(echo "${dfl}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
    # dcp="${OUTPUT_PREFIX:-$(get_output_prefix)}/.cache/${dfl_stem}/index-cache"
    dcp=".cache/${dfl_stem}/index-cache"
    export DATA_FILE_LIST="${dfl}"
    export NUM_DOCS="${ndocs}"
    export WEIGHT_SUM="${ws}"
    export DFL_STEM="${dfl_stem}"
    export DATA_CACHE_PATH="${dcp}"
    export DATA_FLAGS="${DATA_FLAGS} --data-file-list ${DATA_FILE_LIST}"  #  --data-cache-path ${DATA_CACHE_PATH}"
    echo "--------------------"
    echo "Updated environment:"
    printf "DATA_FILE_LIST: %s\n" "${DATA_FILE_LIST}"
    printf "NUM_DOCS: %s\n " "${NUM_DOCS}"
    printf "WEIGHT_SUM: %s\n" "${WEIGHT_SUM}"
    printf "DFL_STEM: %s\n" "${DFL_STEM}"
    printf "DATA_CACHE_PATH: %s\n" "${DATA_CACHE_PATH}"
    printf "DATA_FLAGS: %s\n" "${DATA_FLAGS}"
    echo "--------------------"
    # fi
    # export DATA_FLAGS="${DATA_FLAGS}"
    # export TOKENIZER_FLAGS="${TOKENIZER_FLAGS}"
    # printf "[setData] DATA_FLAGS: %s\n" "$(printGreen ${DATA_FLAGS})"
    # printf "[setData] TOKENIZER_FLAGS: %s\n" "$(printMagenta ${TOKENIZER_FLAGS})"
}

################################################################################
# generateDSconfig: Create and save a deepspeed config .json
#
# This will contain the appropriate variables as set in the current environment.
################################################################################
generateDSconfig() {
    for v in "$GLOBAL_BATCH" "$MICRO_BATCH" "$GRAD_ACC_STEPS" "$ZERO_STAGE" "$PP" "$DTYPE"
    do
      if [ -z "$v" ]; then
        echo "Please export required envs before execute $0"
        exit 1
      fi
    done
    if [ $# -ne 1 ]; then
      echo "Usage: $0 config_file"
      exit 1
    fi
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
    if [ "$ZERO_STAGE" == 3 ]; then
    zero="\
        \"zero_optimization\": {
          \"stage\": 3,
          \"reduce_scatter\": false,
          \"mics_shard_size\": 4,
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
    elif [ "${ZERO_STAGE}" == 2 ] || [ "${ZERO_STAGE}" == 1 ]; then
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
    # elif [[ $ZERO_STAGE == 1 ]]; then
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
      # echo 'please add the config for zero_stage 1 without pipeline-parallelism'
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
# flops_profiler must at the end because no ',' is allowed at the end
cat <<EOT > "$1"
{
$common
$zero
$dtype
$extra
$flops_profiler
}
EOT
}

#####################
# train
#####################
train() {
    # 1. Navigate into `$PBS_O_WORKDIR` <-- [should be Megatron-Deepspeed]
    cd "${PBS_O_WORKDIR}" || exit
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    # 2. source `ALCF/helpers.sh` <-- [should be ./ALCF/helpers.sh]
    source "${HERE}/ALCF/helpers.sh" || exit
    # 3. call `setup` from `./ALCF/helpers.sh`
    export DATA_FILE_LIST="${HERE}/ALCF/data-lists/$(get_machine_name)/books.txt"
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

###############################################
# Helper functions for printing colored text
###############################################
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

export AURORA_GPT_HEADER="""
 █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗        ██████╗ ██████╗ ████████╗
██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗      ██╔════╝ ██╔══██╗╚══██╔══╝
███████║██║   ██║██████╔╝██║   ██║██████╔╝███████║█████╗██║  ███╗██████╔╝   ██║
██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══██║╚════╝██║   ██║██╔═══╝    ██║
██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║  ██║      ╚██████╔╝██║        ██║
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝       ╚═════╝ ╚═╝        ╚═╝
"""

###########################
# call helpers_main()
###########################
helpers_main

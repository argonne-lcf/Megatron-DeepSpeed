#!/bin/bash --login
###############################################################################
# [`ALCF/helpers.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh)
#
# Contains helper functions for launching `../train_alcf.sh`
###############################################################################

###############################################################################
# Source:
# [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
# shellcheck disable=SC1090
source <(curl -L https://bit.ly/ezpz-utils) >/dev/null || return 1
ezpz_setup_job >/dev/null || exit
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
#    - Setup `python`
#       1. Load `conda`
#       2. Setup `venv` on top of `conda`
#    - Ensure all dependencies are installed
#    - Clone + Install [`saforem2/ezpz`](https://github.com/saforem2/ezpz)
#       - Source [`ezpz/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
#           - This provides `{ezpz_setup_python, ezpz_setup_job}` (called below)
#    - Set runtime options
#    - Build `deepspeed_config.json`
#    - Build {logs, checkpoints, etc} dirs, named according to specifics of
#       current run
#    - Specify additional `deepspeed` arguments
#    - Ensure executable exists at expected path
#    - Setup data + tokenizer via `TOKENIZER_TYPE`
#    - Print job info
#    - Save `.env` to `CKPT_DIR` for safe keeping
#    - Check that we're not already running, and if so, exit.
#    - Setup run command to be executed.
##############################################################################
setup() {
	# Identify machine we're on
	mn=$(ezpz_get_machine_name)
	export MACHINE="${mn}"
	# get_machine || exit
	##########################################################################
	# # ezpz_setup will:
	# # 1. Setup python
	# #     - load base conda
	# #     - (if necessary) create virtual environment on top of base conda
	# #     - activate virtual environment from ^
	# # 2. Install ezpz (if needed)
	# # 3. Parse PBS_* environment variables to determine:
	# #     - NHOSTS (by counting number of lines in $PBS_NODEFILE)
	# #     - NGPU_PER_HOST (by magic)
	# #     - NGPUS (= NHOSTS * NGPU_PER_HOST)
	# # 4. Use these (^) to build our launch command
	# ezpz_setup || exit
	##########################################################################
	install_dependencies
	# Set command line arguments to pass to `"${EXEC}"`
	setParams || exit
	# Create `deepspeed_config.json` from runtime params from ^
	buildDSconfig || exit
	# Specify output directory for {logs, checkpoints, etc.}
	setup_checkpoint || exit
	setOutput || exit
	# Specify additional `deepspeed` arguments (dependent on _newly created_ variables)
	set_args || exit
	# Ensure executable exists in expected path
	check_executable "${EXEC:-${WORKING_DIR}/pretrain_gpt_alcf.py}"
	dfl="${DATA_FILE_LIST:-"${PBS_O_WORKDIR:-${HERE}}/ALCF/data-lists/$(ezpz_get_machine_name)/dolma.txt"}"
	# Setup data + tokenizer via `DATA_FILE_LIST` and `TOKENIZER_TYPE`
	tok="${TOKENIZER_TYPE:-HFTokenizer}"
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
	# custom_args="$@"
	custom_args=("$@")
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
	export data_cache_path="${CKPT_DIR}/${DATA_CACHE_PATH}" && mkdir -p "${data_cache_path}"
	printf "\n"
	echo "Using data_cache_path: ${data_cache_path}"
	##################################################################
	# WARN: to disable Llama-type architectures, toggle via:
	# `NO_LLAMA=1 bash train_llama_alcf.sh`
	##################################################################
	if [[ -z "${NO_LLAMA:-}" ]]; then
		llama_flags=(
			"--swiglu"
			"--hidden-dropout 0"
			"--attention-dropout 0"
			"--normalization rmsnorm"
			"--disable-bias-linear"
			"--no-query-key-layer-scaling"
			"--use-rotary-position-embeddings"
			"--untie-embeddings-and-output-weights"
			"--num-key-value-heads ${NUM_KV_HEAD}"
			"--ffn-hidden-size ${FFN_HIDDEN_SIZE}"
		)
	fi

	tb_flags=()
	if [[ -z "${NO_TENSORBOARD:-}" ]]; then
		TBDIR="${CKPT_DIR}/tensorboard"
		mkdir -p "${TBDIR}"
		tb_flags+=(
			"--log-timers-to-tensorboard"
			"--log-optimizer-states-to-tensorboard"
			"--tensorboard-dir ${TBDIR}"
		)
	fi
	dfl_fallback="${DATA_FILE_LIST:-${PBS_O_WORKDIR}/ALCF/data-lists/$(ezpz_get_machine_name)/dolma.txt}"

	train_args=()
	if [[ -z "${OVERRIDE_CKPT_OPT_PARAM:-}" ]]; then
		train_args+=("--use-checkpoint-opt_param-scheduler")
	fi
	# "--init-method-std ${INIT_METHOD_STD:-0.0006}"
	# "--shuffle-sample"
	train_args+=(
		"${lr_flags[@]}"
		"${custom_args[@]}"
		"${llama_flags[@]}"
		"${FLASH_ARG}"
		"${TIMING_STR:-}"
		"${DATA_FLAGS}"
		"${TOKENIZER_FLAGS}"
		"${tb_flags[@]}"
		"${ds_args[@]}"
		"--${DTYPE}"
		"--accumulate-allreduce-grads-in-fp32"
		"--adjust-word-embedding-init"
		"--adam-beta1=${ADAM_BETA1:-0.9}"
		"--adam-beta2=${ADAM_BETA2:-0.95}"
		"--adam-eps=${ADAM_EPS:-0.00001}"
		# "--blend-sample-in-corpus"
		"--clip-grad=${CLIP_GRAD:-1.0}"
		"--data-cache-path=${data_cache_path}"
		"--data-file-list=${DATA_FILE_LIST:-${dfl_fallback}}"
		"--distributed-backend=${BE}"
		"--ds-sequence-parallel-size=${SP}"
		"--eval-interval=${EVAL_INTERVAL:-100}"
		"--eval-iters=${EVAL_ITERS:-20}"
		"--global-batch-size=${GLOBAL_BATCH}"
		"--hidden-size=${HIDDEN}"
		"--init-method-std=$(echo "scale=6; sqrt(2 / (5 * ${HIDDEN}))" | bc -l)"
		"--log-interval=${LOG_INTERVAL:-1}"
		"--load=${LOAD:-${CKPT_DIR}}"
		"--max-position-embeddings=$((16 * SEQ))"
		"--micro-batch-size=${MICRO_BATCH}"
		"--no-bias-gelu-fusion"
		"--no-bias-dropout-fusion"
		"--no-masked-softmax-fusion"
		"--no-gradient-accumulation-fusion"
		"--num-layers=${NLAYERS}"
		"--num-attention-heads=${HEADS}"
		"--optimizer=${OPT}"
		"--pipeline-model-parallel-size=${PP}"
		"--rotary-position-embeddings-theta=${ROPE_THETA:-50000}"
		"--save=${SAVE:-${CKPT_DIR}}"
		"--seq-length=${SEQ}"
		"--split=${TRAIN_SPLIT:-990},${VAL_SPLIT:-10},${TEST_SPLIT:-0}"
		"--shuffle-sample-in-corpus"
		"--save-interval=${SAVE_INTERVAL:-50}"
		"--train-iters=${TRAIN_ITERS}"
		"--tensor-model-parallel-size=${TP}"
		"--timing-log-level=${TIMING_LOG_LEVEL:-1}"
		"--weight-decay=${WEIGHT_DECAY:-0.1}"
		"--word-embedding-init-std=0.632455532"
	)
	declare -A arch_map
	printf "==== ARCHITECTURE ====\n"
	arch_map=(
		["MODEL_ARCH"]="${MODEL_ARCH}"
		["TP"]="${TP}"
		["PP"]="${PP}"
		["SP"]="${SP}"
		["DP"]="${DP}"
		["ZERO"]="${ZERO_STAGE}"
		["MBS"]="${MICRO_BATCH}"
		["GAS"]="${GRAD_ACC_STEPS}"
		["GBS"]="${GLOBAL_BATCH}"
		["NLAYERS"]="${NLAYERS}"
		["HIDDEN"]="${HIDDEN}"
		["HEADS"]="${HEADS}"
		["SEQ"]="${SEQ}"
		["FFN_HIDDEN_SIZE"]="${FFN_HIDDEN_SIZE}"
		["NUM_KV_HEAD"]="${NUM_KV_HEAD}"
		["USE_ACTIVATION_CHECKPOINTING"]="${USE_ACTIVATION_CHECKPOINTING:-0}"
		["DTYPE"]="${DTYPE}"
		["OPT"]="${OPT}"
		# ["WEIGHT_DECAY"]="${WEIGHT_DECAY}"
		# ["ADAM_BETA1"]="${ADAM_BETA1}"
		# ["ADAM_BETA2"]="${ADAM_BETA2}"
		# ["ADAM_EPS"]="${ADAM_EPS}"
		# ["CLIP_GRAD"]="${CLIP_GRAD:-1.0}"
	)
	for v in "${!arch_map[@]}"; do
		printf "%s: %s\n" "$(printGreen "${v}")" "$(printBlue "${arch_map[${v}]}")"
	done
	printf "======================\n"
	# "--adam-eps ${ADAM_EPS:-0.00001}"
	cache_dir="${PBS_O_WORKDIR}/.cache/"
	mkdir -p "${cache_dir}"
	targs_cache="${cache_dir}/train_args.txt"
	for arg in "${train_args[@]}"; do echo "${arg}" >>"${targs_cache}"; done
	export TRAIN_ARGS=("$(printf '%s\n' "${train_args[@]}" | sort)")
	printf "Training Arguments: %s\n" "$(printBlue "${TRAIN_ARGS[@]}")"
	export run_cmd=("${LAUNCHER}" "${train_args[@]}")
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
	echo "- MPICH_DIR=${MPICH_DIR:-${MPI_ROOT:-}}"
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
	shell_type=$(basename "${SHELL}")
	if [[ "${shell_type}" == "bash" ]]; then
		shopt -s expand_aliases
	fi
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
    LAUNCHER="ezpz-launch $(which python3) ${EXEC}"
		# source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_job
		# echo "ezpz_launch: $(which ezpz_launch)"
		# export -f ezpz_launch
		# export LAUNCHER="ezpz_launch) $(which python3) -Wignore ${EXEC}"
		# if [[ -n "${DIST_LAUNCH}" ]]; then
		#     mn=$(ezpz_get_machine_name)
		#     if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
		#         LAUNCHER="${DIST_LAUNCH} --pmi=pmix --genvall $(which python3) -Wignore ${EXEC}"
		#     elif [[ "${mn}" == "sophia" ]]; then
		#         LAUNCHER="${DIST_LAUNCH} $(which python3) -Wignore ${EXEC}"
		#     else
		#         LAUNCHER="${DIST_LAUNCH} --genvall $(which python3) -Wignore ${EXEC}"
		#     fi
		#     export LAUNCHER="${LAUNCHER}"
		# else
		#     echo "[setupLauncher][INFO]: Saving environment to: .env-${PBS_JOBID}"
		#     printenv | tee ".env-${PBS_JOBID}"
		#     echo "[setupLauncher][ERROR]: DIST_LAUNCH not found in environment !!"
		# fi
	fi
	printf "Launching with: %s\n" "$(printRed "${dist_launcher}")"
	printf " %s" "$(printMagenta "${LAUNCHER}")"
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
# |     nnhosts     |   nhosts   |  GAS  |
# |:---------------:|:----------:|:-----:|
# | 256 <= n < inf  | [256, inf) |   1   |
# | 128 <= n < 256  | [128, 256) |   2   |
# |  32 <= n < 128  | [32, 128)  |   4   |
# |  16 <= n < 32   | [16, 32)   |   8   |
# |   0 <= n < 16   | [0, 16)    |  16   |
#
###########################################
get_grad_acc_steps_on_aurora() {
	if [[ "$#" == 0 ]]; then
		hf="${HOSTFILE:-${PBS_NODEFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}}"
	elif [[ "$#" == 1 ]]; then
		hf="$1"
	else
		echo "Usage: get_grad_acc_steps_on_aurora"
		echo "Expected exactly 0 or 1 arguments, received: $#"
		exit 1
	fi
	nhosts=$(wc -l <"${hf}")
	if [[ "${nhosts}" -ge 256 ]]; then #   n >= 256
		gas=1
	elif [[ 128 -le "${nhosts}" && "${nhosts}" -lt 256 ]]; then # 128 <= n < 256
		gas=2
	elif [[ 32 -lt "${nhosts}" && "${nhosts}" -lt 129 ]]; then #  32 < n  < 128
		gas=4
	elif [[ 16 -le "${nhosts}" && "${nhosts}" -le 32 ]]; then #  16 <= n < 32
		gas=8
	else
		gas=16
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
	# export CCL_ATL_TRANSPORT=mpi     # Required by Aurora mpich
	export CCL_ATL_TRANSPORT=ofi # [SF]: Changed 04/30/2025
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


get_model_arch_AuroraGPT_2B() {
	# AuroraGPT-2B
	export HEADS=16
	export NLAYERS=12
	export HIDDEN=2048
	export NUM_KV_HEAD=4
	export FFN_HIDDEN_SIZE=11008
	export SEQ=8192
	export MODEL_ARCH="AuroraGPT-2B"
}

get_model_arch_7B() {
	# 7B
	# export MODEL_KEY="AuroraGPT-7B"
	export HEADS=${HEADS:-${NHEADS:-32}}             # NUMBER OF ATEN HEADS
	export NLAYERS=${NLAYERS:-${NUM_LAYERS:-32}}     # NUMBER OF LAYERS
	export HIDDEN=${HIDDEN:-4096}                    # HIDDEN SIZE
	export NUM_KV_HEAD=${NUM_KV_HEAD:-8}             # GROUP ATTENTION
	export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008} # FFN HIDDEN SIZE
	export SEQ=${SEQ:-4096}                          # SEQ_LEN: 4096
	export MODEL_ARCH="AuroraGPT-7B"
}

get_model_arch_llama3_3B() {
	export HEADS=24
	export NLAYERS=28
	export HIDDEN=3072
	export NUM_KV_HEAD=8
	export FFN_HIDDEN_SIZE=8192
	export SEQ=8192
	export MODEL_ARCH="llama3-3B"
}

get_model_arch_smollm3_3B() {
	export HEADS=16
	export NLAYERS=36
	export HIDDEN=2048
	export NUM_KV_HEAD=4
	export FFN_HIDDEN_SIZE=11008
	export SEQ=8192
	export MODEL_ARCH="smollm3-3B"
}

get_model_arch_phi4_mini() {
	export HEADS=32
	export NLAYERS=24
	export HIDDEN=3072
	export NUM_KV_HEAD=8
	export FFN_HIDDEN_SIZE=8192
	export SEQ=8192
	export MODEL_ARCH="phi4-mini"
}

get_model_arch_llama3_3B_customNlayers() {
	export HEADS=24
	export NLAYERS="${NLAYERS:-28}" # default to 28 layers
	export HIDDEN=3072
	export NUM_KV_HEAD=8
	export FFN_HIDDEN_SIZE=8192
	export SEQ=8192
	export MODEL_ARCH="llama3-3B-nLayers${NLAYERS}"
}

get_model_arch_smollm3_3B_custom_nLayers() {
	export HEADS=16
	export NLAYERS="${NLAYERS:-24}" # default to 24 layers
	export HIDDEN=2048
	export NUM_KV_HEAD=4
	export FFN_HIDDEN_SIZE=11008
	export SEQ=8192
	export MODEL_ARCH="smollm3-nLayers${NLAYERS}"
}

get_model_arch_phi4_mini_custom_nLayers() {
	export HEADS=32
	export NLAYERS="${NLAYERS:-24}" # default to 24 layers
	export HIDDEN=3072
	export NUM_KV_HEAD=8
	export FFN_HIDDEN_SIZE=8192
	export SEQ=8192
	export MODEL_ARCH="phi4-mini-nLayers${NLAYERS}"
}

# get_model_arch_70B() {
#     # 70B
#     export MODEL_KEY="AuroraGPT-70B"
#     export TP=6
#     export ZERO_STAGE=2
#     export GRAD_ACC_STEPS=1
#     export USE_ACTIVATION_CHECKPOINTING=1
#     export MICRO_BATCH=1
#     export NLAYERS=80
#     export HIDDEN=8192
#     export HEADS=96
#     export FFN_HIDDEN_SIZE=11040
#     export SEQ=8192
#     export NUM_KV_HEAD=6
# }
#
get_model_arch_70B() {
	HEADS=64
	NLAYERS=80
	NUM_KV_HEAD=8
	FFN_HIDDEN_SIZE=28672
	HIDDEN=8192
	SEQ=8192
	export MODEL_ARCH="AuroraGPT-70B"
}

get_model_arch_33B() {
	# 33B
	export MODEL_KEY="AuroraGPT-33B"
	export TP=6
	export ZERO_STAGE=2
	export GRAD_ACC_STEPS=1
	# export USE_ACTIVATION_CHECKPOINTING=1
	export MICRO_BATCH=1
	export NLAYERS=78
	export HIDDEN=8268
	export HEADS=78
	export FFN_HIDDEN_SIZE=11076
	export SEQ=4096
	export NUM_KV_HEAD=6
	export MODEL_ARCH="AuroraGPT-33B"
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
	# ---- [Parallelism Settings] -------------------------------------------+
	# ------ [Aurora] -------||------ [SunSpot] -------------
	# if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
	GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH
	mn=$(ezpz_get_machine_name)
	if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
		TP=${TP:-1} # TP = 1
		export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
		export CCL=${CCL:-ccl}      # CCL
		export BE="${CCL}"          # COMMUNICATION BACKEND = CCL
		export DTYPE=${DTYPE:-bf16} # DTYPE: bf16
		# export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}     # GRADIENT_ACC_STEPS
		gas=$(get_grad_acc_steps_on_aurora "${PBS_NODEFILE:-${HOSTFILE:-${hostfile}}}")
		export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-${gas}}"
		# export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-$(get_grad_acc_steps_on_aurora "$@)}"
		echo "[setParams] Using GRAD_ACC_STEPS: ${GRAD_ACC_STEPS}"
		MICRO_BATCH=${MICRO_BATCH:-1}
		if [[ -n "${NO_FLASH_ATTN-}" ]]; then
			echo "Not using flash-attn!!"
		else
			FLASH_ARG="--use-flash-attn-builder"
		fi
	# [Polaris]
	elif [[ "${mn}" == "polaris" || "${mn}" == "sirius" ]]; then
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
		if [[ -n "${NO_FLASH_ATTN:-}" ]]; then
			echo "Not using flash-attn!!"
		else
			FLASH_ARG="--use-flash-attn-v2"
		fi
		echo "Setting up AWS NCCL OFI Plugin on Polaris..."
		source "${WORKING_DIR}/ALCF/aws_ofi_nccl_plugin.sh" || exit
	# ---- [Sophia] ----------------------
	elif [[ "${mn}" == sophia* ]]; then
		# export LAUNCH_CMD="${LAUNCH_CMD:-deepspeed}"
		TP=${TP:-1}                                # TP = 2
		export NCCL=${NCCL:-nccl}                  # NCCL
		export BE="${NCCL}"                        # BE = NCCL
		export DTYPE=${DTYPE:-bf16}                # DTYPE: FP16
		export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8} # GRADIENT_ACC_STEPS
		export MICRO_BATCH="${MICRO_BATCH:-$(get_batch_size_on_polaris)}"
		if [[ -n "${NO_FLASH_ATTN-}" ]]; then
			echo "Not using flash-attn!!"
		else
			FLASH_ARG="--use-flash-attn-v2"
		fi
		# echo "Setting up AWS NCCL OFI Plugin on Polaris..."
		# source "${WORKING_DIR}/ALCF/aws_ofi_nccl_plugin.sh" || exit
	# [Perlmutter]
	elif [[ "${mn}" == login* || "${mn}" == nid* ]]; then
		TP="${TP:-2}"
		export NCCL="${NCCL:-nccl}"
		export BE="${NCCL}"
		export DTYPE="${DTYPE:-bf16}"
		MICRO_BATCH="${MICRO_BATCH:-1}"
		if [[ -n "${NO_FLASH_ATTN-}" ]]; then
			echo "Not using flash-attn!!"
		else
			FLASH_ARG="--use-flash-attn-v2"
		fi
	fi
	ma="${MODEL_ARCH:-7B}"
	case "${ma}" in
	# "70B" | "llama-3.1-70B" | "llama-3.1-70b" | "llama-3.2-70B" | "llama-3.2-70b")
	"70B")
		get_model_arch_70B
		;;
	"33B" | "llama-3.2-33B" | "llama-3.2-33b")
		get_model_arch_33B
		;;
	"smollm3-3B" | "smollm3_3B")
		get_model_arch_smollm3_3B_custom_nLayers
		;;
	"phi4-mini" | "phi4_mini")
		get_model_arch_phi4_mini_custom_nLayers
		;;
	"llama3-3B" | "llama-3B")
		get_model_arch_llama3_3B_customNlayers
		;;
  "2B" | "AuroraGPT-2B" | "AuroraGPT_2B" | "Aurora-GPT-2B" | "AuroraGPT2B" | "Aurora_GPT_2B" | "aurora-gpt-2b" | "aurora_gpt_2b")
    get_model_arch_AuroraGPT_2B
    ;;
	"7B" | "AuroraGPT-7B" | "aurora-gpt-7b" | "llama-3.1-7B" | "llama-3.1-7b" | "llama-3.2-7B" | "llama-3.2-7b")
		get_model_arch_7B
		;;
	*)
		get_model_arch_7B
		;;
	esac
	export TP="${TP}"
	export PP="${PP:-1}"
	export SP="${SP:-1}"
	export FLASH_ARG="${FLASH_ARG}"
	export DTYPE="${DTYPE:-bf16}"
	OPT="${OPT:-${OPTIMIZER:-${OPTIM:-adamw}}}"
	export OPT
	log_message INFO "❗ Using OPT: ${OPT}"
	export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
	export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
	NHOSTS=$(wc -l <"${HOSTFILE}")
	if [[ -z "${NGPU_PER_HOST:-}" ]]; then
		NGPU_PER_HOST=$(python3 -c 'import ezpz as ez; print(ez.get_gpus_per_node())')
	fi
	export NGPU_PER_HOST="${NGPU_PER_HOST}"
	export WORLD_SIZE="${WORLD_SIZE:-$((NHOSTS * NGPU_PER_HOST))}"
	# if [[ "${WORLD_SIZE}" -gt 1 && "${mn}" == "aurora" ]]; then
	#     #### [sam: 08/17/2024] ##########################################
	#     # Use best set of CCL env vars from Gordon Bell runs on Aurora
	#     set_ccl_vars_on_aurora
	# fi
	# + --[LR Settings]------------------------------------------------------+
	export LR="${LR:-0.0002}"
	export LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
	export LR_WARMUP_FRAC="${LR_WARMUP_FRAC:-0.05}"
	lr_flags=(
		"--lr ${LR}"
		"--lr-decay-style ${LR_DECAY_STYLE}"
		"--lr-warmup-fraction ${LR_WARMUP_FRAC}"
	)
	if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
		lr_flags+=("--lr-decay-iters ${LR_DECAY_ITERS:-}")
	fi
	# +---[Run Settings]------------------------------------------------------+
	export ZERO_STAGE=${ZERO_STAGE:-1}                                                    # ZERO OFFLOADING STAGE
	export MICRO_BATCH=${MICRO_BATCH:-1}                                                  # MICRO BATCH SIZE
	export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}                                            # GRADIENT ACCUMULATION STEPS
	export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"                                      # TIMING VERBOSITY IN LOGS
	export ACT_CKPT_NUM_LAYERS="${ACT_CKPT_NUM_LAYERS:-1}"                                # NUM LAYERS TO CHECKPOINT ACTIVATIONS
	export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-}                 # USE ACTIVATION CHECKPOINTING ?
	export GLOBAL_BATCH_MAX=$((WORLD_SIZE * MICRO_BATCH * GRAD_ACC_STEPS / TP / PP / SP)) # MAX GLOBAL BATCH SIZE
	export DP=$((WORLD_SIZE / TP / PP / SP))                                              # DATA PARALLELISM
	export GLOBAL_BATCH="${GLOBAL_BATCH:-${GLOBAL_BATCH_MAX}}"                            # WILL USE MAX IF NOT SET IN ENVIRONMENT
	echo "WORLD_SIZE: ${WORLD_SIZE} MBS: ${MICRO_BATCH} GAS: ${GRAD_ACC_STEPS} DP: ${DP} GBS: ${GLOBAL_BATCH} GBS_MAX: ${GLOBAL_BATCH_MAX} TP: ${TP} PP: ${PP} SP: ${SP}"
	if [[ -n "${TRAIN_TOKENS:-}" ]]; then
		export TRAIN_TOKENS="${TRAIN_TOKENS}"
		export TRAIN_ITERS=$((TRAIN_TOKENS / SEQ / GLOBAL_BATCH))
		printf "TRAIN_TOKENS=%s (=%sB tokens)\n" "${TRAIN_TOKENS}" "$((TRAIN_TOKENS / 10 ** 9))"
		printf "TRAIN_ITERS=%s\n" "${TRAIN_ITERS}"
	elif [[ -z "${TRAIN_ITERS:-${TRAIN_ITER:-}}" ]]; then
		export TRAIN_TOKENS=${TRAIN_TOKENS:-4673780159710}
		export TRAIN_ITERS=$((TRAIN_TOKENS / SEQ / GLOBAL_BATCH))
		printf "TRAIN_TOKENS=%s (=%sB tokens)\n" "${TRAIN_TOKENS}" "$((TRAIN_TOKENS / 10 ** 9))"
		printf "TRAIN_ITERS=%s\n" "${TRAIN_ITERS}"
	else
		export TRAIN_ITERS="${TRAIN_ITERS:-${TRAIN_ITER:-}}"
	fi
	export MODEL_TYPE="${MODEL_ARCH:-AuroraGPT}-gb${GLOBAL_BATCH}-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden" # STRING FOR IDENTIFYING MODEL
	# NOTE: [2024-07-10] #####################################################
	# - [sam]: For whatever reason, it seems that using
	#   sequence-parallelism (SP) > 1 is INCOMPATIBLE with
	#   rotary-position-embeddings (ROPE).
	#
	#   For this reason, we only use the default LLAMA_ARGS when SP=0.
	##########################################################################
	# # -----[Learning Rate Settings]--------------------------------------------
	# export LR=${LR:-0.0002}                       # LEARNING_RATE
	# export LR_WARMUP_FRAC=${LR_WARMUP_FRAC:-0.05} # LEARNING RATE WARMUP
	# export LR_DECAY_ITERS=${LR_DECAY_ITERS:-}     # LR DECAY ITERS
	# set_lr_args
	# -----[Learning Rate Settings]--------------------------------------------
	# # if [[ "${TIMING_LOG_LEVEL:-1}" -gt 1 ]]; then
	# if [[ "${TIMING_LOG_LEVEL:-1}" -gt 1 ]]; then
	#     TIMING_STR="\
	#         --timing-log-level ${TIMING_LOG_LEVEL}"
	#     # "
	# else
	#     TIMING_STR=""
	# fi
}

##############################################
# set_args
#
# Specify additional (DeepSpeed specific)
# arguments to pass to pretrain_gpt_alcf.py
##############################################
set_args() {
	# ---- Set DeepSpeed arguments --------------------------------
	ds_args=(
		"--deepspeed"
	)
	if [[ "${PP:-1}" == 1 ]]; then
		ds_args+=("--no-pipeline-parallel")
	fi
	ds_args+=("--deepspeed_config=${DS_CONFIG}")
	ds_args+=("--zero-stage=$ZERO_STAGE")
	# if [[ -n "${USE_ACTIVATION_CHECKPOINTING:-}" ]]; then
	if [[ "${USE_ACTIVATION_CHECKPOINTING:-}" == 1 || "${USE_ACTIVATION_CHECKPOINTING:-}" == "true" ]]; then
		echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
		ds_args+=("--deepspeed-activation-checkpointing")
		ds_args+=(
			"--checkpoint-activations"
			"--checkpoint-num-layers=${ACT_CKPT_NUM_LAYERS:-1}"
		)
		# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
		# --checkpoint-activations \
		# --deepspeed-activation-checkpointing
	fi
	export ds_args
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
		echo "PATH=${PATH:-}"
		echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
		echo "http_proxy=${http_proxy:-}"
		echo "https_proxy=${https_proxy:-}"
		echo "CFLAGS=${CFLAGS:-}"
		echo "PYTHONUSERBASE=${PYTHONUSERBASE:-}"
	} >.deepspeed_env
}

get_output_prefix() {
	# ---- Specify output location --------------------------------
	pre="ws${WORLD_SIZE}-ds-stage${ZERO_STAGE}-nl${NLAYERS}"
	pre="${pre}-hs${HIDDEN}-mb${MICRO_BATCH}"
	pre="${pre}-seq${SEQ}-gb${GLOBAL_BATCH}"
	pre="${pre}-sp${SP}-pp${PP}-tp${TP}-${DTYPE}-opt${OPT}"
	pre="${pre}-lr${LR}-lwf${LR_WARMUP_FRAC}"
	pre="${MODEL_ARCH:-AuroraGPT}-${pre}"
	local num_tokens_in_billions
	num_tokens_in_billions=$((TRAIN_TOKENS / 10 ** 9))
	pre="${pre}_ntok${num_tokens_in_billions}B"
	if [[ -n "${TOKENIZER_TYPE:-}" ]]; then
		# _tok="${TOKENIZER_TYPE/Tokenizer//}" # Strip "Tokenizer" suffix if present
		_tok=$(echo "${TOKENIZER_TYPE}" | sed 's/Tokenizer//g') # noqa
		pre="${pre}_tok${_tok}"
	fi
	if [[ -n "${TOKENIZER_MODEL:-}" ]]; then
		# _tm=$(echo "${TOKENIZER_MODEL}" | sed 's/\/_/g') # noqa
		# replace slashes with underscores
		_tm="${TOKENIZER_MODEL//\//_}" # noqa
		pre="${pre}_tm${_tm}"
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
	echo "${OUTPUT_LOG}" >>"logs/latest"
	printf "\n Please see logs at: %s\n" "$(printGreen "${OUTPUT_DIR}")"
}

get_checkpoint_dir() {
	if [[ -n "${CKPT_DIR:-}" ]]; then
		echo "${CKPT_DIR}"
	else
		echo "checkpoints/$(get_output_prefix)"
	fi
}

setup_checkpoint() {
	ckpt_dir=$(get_checkpoint_dir)
	export CKPT_DIR="${ckpt_dir}"
	printf "Checkpoints will be saved to: %s\n" "$(printYellow "${CKPT_DIR}")"
}

#############################################
# Build DeepSpeed config and write to .json
#############################################
buildDSconfig() {
	# export CPU_OPTIMIZER="${CPU_OPTIMIZER:-0}"
	export DS_CONFIG="${WORKING_DIR}/ds-configs/ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
	mkdir -p "$(dirname "${DS_CONFIG}")"
	printf "DS_CONFIG: %s\n" "${DS_CONFIG}"
	printf "ZS=%s, MB=%s, GB=%s, PP=%s, DTYPE=%s\n" "${ZERO_STAGE}" "${MICRO_BATCH}" "${GLOBAL_BATCH}" "${PP}" "${DTYPE}"
	generateDSconfig "${DS_CONFIG}"
	cat "${DS_CONFIG}" | jq .
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
	python3 -m pip install -r "${depsfile}" # --require-virtualenv
	if [[ ! -x "$(command -v deepspeed)" ]]; then
		printf "[install_dependencies] No 'deepspeed' command found on %s in %s\n" "$$(ezpz_get_machine_name)" "$(which python3)"
		printf "[install_dependencies] Attempting to install deepspeed via pip...\n"
		python3 -m pip install deepspeed || { # --require-virtualenv || {
			printf "[install_dependencies] Failed to install deepspeed via pip on %s\n" "$(ezpz_get_machine_name)"
			# printf "[install_dependencies] !! No deepsepeed in %s\n" "$(which python3)"
			return 1
		}
		# mn=$(ezpz_get_machine_name)
		# if [[ "${mn}" == aurora* || "${mn}" == sunspot* ]]; then
		#     install_deepspeed_for_xpu || exit
		# fi
		# printf "[install_dependencies] !! No deepsepeed in %s" "$(which python3)"
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
	_data_flags=()
	_tokenizer_flags=()
	if [[ ${tok} == gpt* || ${tok} == GPT* ]]; then
		export TOKENIZER_TYPE="GPT2"
		_tokenizer_flags+=("--tokenizer-type GPT2BPETokenizer")
		machine=$(ezpz_get_machine_name)
		if [[ ${machine} == "polaris" || ${machine} == "sophia" ]]; then
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
		_data_flags+=(
			"--data-path ${DATA_PATH}"
			"--vocab-file ${VOCAB_FILE}"
			"--merge-file ${MERGE_FILE}"
		)
	else
		# export TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
		# tm="${WORKING_DIR}/ALCF/tokenizer.model"           # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
		export TOKENIZER_TYPE="${TOKENIZER_TYPE:-HFTokenizer}"
		tm="${TOKENIZER_MODEL:-google/gemma-7B}"           # fallback: Megatron-DeepSpeed/ALCF/tokenizer.model
		export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${tm}}" # USE TOKENIZER_MODEL from env, else fallback from ^
		_tokenizer_flags+=(
			"--tokenizer-type ${TOKENIZER_TYPE}"
			"--tokenizer-model ${TOKENIZER_MODEL}"
		)
		# if [[ "${TOKENIZER_TYPE}" != "GPT2" ]]; then
		echo "Using tokenizer: ${TOKENIZER_TYPE}. Setting up data with ${dfl}"
		setData "${dfl}" || exit
	fi
	export DATA_FLAGS="${_data_flags[*]:-}"
	export TOKENIZER_FLAGS="${_tokenizer_flags[*]}"
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
	mn=$(ezpz_get_machine_name)
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
	# export DATA_FLAGS="${DATA_FLAGS} --data-file-list ${DATA_FILE_LIST}"   #  --data-cache-path ${DATA_CACHE_PATH}"
	echo "--------------------"
	echo "Updated environment:"
	printf "DATA_FILE_LIST: %s\n" "${DATA_FILE_LIST}"
	printf "NUM_DOCS: %s\n " "${NUM_DOCS}"
	printf "WEIGHT_SUM: %s\n" "${WEIGHT_SUM}"
	printf "DFL_STEM: %s\n" "${DFL_STEM}"
	printf "DATA_CACHE_PATH: %s\n" "${DATA_CACHE_PATH}"
	printf "DATA_FLAGS: %s\n" "${DATA_FLAGS:-}"
	echo "--------------------"
}

generateDSconfig_new() {
	cat <<EOT >"${CONFIG_JSON}"
    {
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1,

    "zero_optimization": {
        "stage": $ZERO_STAGE
    },

    "bf16": {
        "enabled": true
    },

    "data_types": {
            "grad_accum_dtype": "fp32" 
    },

    "wall_clock_breakdown" : false
    }
EOT
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
        \"gradient_clipping\": 1.0,
        \"steps_per_print\": 1,
        \"gradient_accumulation_steps\": $GRAD_ACC_STEPS,
        \"zero_force_ds_cpu_optimizer\": false,
        \"zero_allow_untested_optimizer\": true,
        \"wall_clock_breakdown\": false,"
	# if [[ "${USE_ACTIVATION_CHECKPOINTING}" == 1 ]]; then
	#     activation_checkpointing="\
	#         \"activation_checkpointing\": {
	#         \"partition_activations\": true,
	#         \"contiguous_memory_optimization\": true
	#         },"
	# fi
	if [[ $DTYPE == "bf16" ]]; then
		# \"communication_data_type\": \"bf16\",
		dtype="\
            \"fp16\": {
              \"enabled\": false,
              \"loss_scale\": 0,
              \"loss_scale_window\": 1000,
              \"hysteresis\": 2,
              \"min_loss_scale\": 1
            },
            \"bfloat16\": {
              \"enabled\": true
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
	if [[ "${OPT:-}" == "ds.adamw" ]]; then
		optimizer="\
            \"optimizer\": {
                \"type\": \"AdamW\",
                \"params\": {
                \"lr\": ${LR},
                \"beta1\": ${ADAM_BETA1},
                \"beta2\": ${ADAM_BETA2},
                \"eps\": ${ADAM_EPS},
                \"weight_decay\": 1e-1
            },
        },"
	elif [[ "${OPT:-}" == "ds.onebitlamb" ]]; then
		optimizer="\
            \"optimizer\": {
                \"type\": \"OneBitLamb\",
                \"params\": {
                    \"lr\": 11e-3,
                    \"max_coeff\": 0.3,
                    \"min_coeff\": 0.01,
                    \"freeze_step\": 1000,
                    \"cuda_aware\": false,
                    \"comm_backend_name\": \"${BE}\",
                    \"coeff_beta\": 0.9,
                    \"factor_max\": 4.0,
                    \"factor_min\": 0.5,
                    \"factor_threshold\": 0.1
                }
            },"
	else
		optimizer=""
	fi
	if [[ "${ZERO_STAGE}" == 3 ]]; then
		# \"mics_shard_size\": 2,
		# hpz_partition_size=${ZERO_HPZ_PARTITION_SIZE:-${NGPU_PER_HOST:-1}}
		# zero="\
		#     \"zero_optimization\": {
		#         \"stage\": 3,
		#         \"reduce_scatter\": true,
		#         \"reduce_bucket_size\": 10000000,
		#         \"reduce_scatter\": true,
		#         \"zero_quantized_weights\": true,
		#         \"zero_hpz_partition_size\": $hpz_partition_size,
		#         \"zero_quantized_gradients\": true,
		#         \"contiguous_gradients\": true,
		#         \"overlap_comm\": true,
		#     }
		# },"
		# mics_shard_size="${MICS_SHARD_SIZE:-${NGPU_PER_HOST}}"
		# \"mics_shard_size\": $mics_shard_size,
		hpz_partition_size=${ZERO_HPZ_PARTITION_SIZE:-1}

		# \"zero_quantized_weights\": false,
		# \"zero_hpz_partition_size\": $hpz_partition_size,
		# \"zero_quantized_gradients\": false,
		# \"mics_shard_size\": 1,
		# \"mics_hierarchical_params_gather\": false,
		zero="\
            \"zero_optimization\": {
              \"stage\": 3,
              \"zero_hpz_partition_size\": $hpz_partition_size,
              \"reduce_scatter\": false,
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
		if [[ -n "${CPU_OPTIMIZER:-}" ]]; then
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
                \"enabled\": ${COMMS_LOGGER:-false},
                \"verbose\": false,
                \"debug\": false
              },"
		fi
	else
		echo 'Please add the correct config set!!!'
	fi
	flops_profiler="\
        \"flops_profiler\": {
          \"enabled\": true,
          \"profile_step\": 2,
          \"module_depth\": -1,
          \"top_modules\": 1,
          \"detailed\": true,
          \"output_file\": null
        }"
	cat <<EOT >"$1"
{
$common
$optimizer
$zero
$dtype
$extra
$flops_profiler
}
EOT
}

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

reset_env() {
	custom_vars=(
		NO_FLASH_ATTN
		USE_FLASH_ATTN
		TP
		PP
		SP
		FLASH_ARG
		OPT
		ADAM_BETA1
		ADAM_BETA2
		ADAM_EPS
		WEIGHT_DECAY
		HEADS
		NLAYERS
		HIDDEN
		NUM_KV_HEAD
		FFN_HIDDEN_SIZE
		SEQ
		ZERO_STAGE
		MICRO_BATCH
		EVAL_ITERS
		EVAL_INTERVAL
		TIMING_LOG_LEVEL
		ACT_CKPT_NUM_LAYERS
		USE_ACTIVATION_CHECKPOINTING
		GLOBAL_BATCH_MAX
		GLOBAL_BATCH
		TRAIN_TOKENS
		TRAIN_ITERS
		MODEL_TYPE
		LR
		LR_WARMUP_FRAC
		LR_DECAY_ITERS
		LR_ARGS
		CPU_OPTIMIZER
		DS_CONFIG
		OUTPUT_DIR
		OUTPUT_LOG
		CKPT_DIR
		ds_args
		EXEC
		EXEC_STEM
		DATA_FLAGS
		TOKENIZER_TYPE
		TOKENIZER_MODEL
		TOKENIZER_FLAGS
		DATA_FILE_LIST
		NUM_DOCS
		WEIGHT_SUM
		DFL_STEM
		DATA_CACHE_PATH
		DOTENV_FILE
		YEAR
		MONTH
		DAY
		TODAY
		STARTED_AT
		LAUNCHER
		data_cache_path
		DEFAULTS
	)
	# LLAMA_ARGS
	printf "Unsetting custom vars: %s\n" "${custom_vars[*]}"
	unset "${custom_vars[@]}"
}

convert_ckpt_to_universal() {
	if [[ "$#" -ne 1 ]]; then
		echo "Usage: convert_ckpt_to_universal ckpt_dir"
		echo "Expected one argument (ckpt_dir), received: $#"
		exit 1
	fi
	ckptdir=$1
	gs=$(cat "${ckptdir}/latest_checkpointed_iteration.txt")
	src="${ckptdir}/global_step${gs}"
	dst="${ckptdir}/global_step${gs}_universal"
	convert_script="${PBS_O_WORKDIR}/deps/DeepSpeed/checkpoint/ds_to_universal.py"
	python3 "${convert_script}" --input_folder "${src}" --output_folder "${dst}"
}

###########################
# call helpers_main()
###########################
helpers_main

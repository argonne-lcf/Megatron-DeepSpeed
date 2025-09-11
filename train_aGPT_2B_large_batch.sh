#!/bin/bash --login
#PBS -q prod
#PBS -A AuroraGPT
#PBS -j oe
#PBS -l walltime=06:00:00,filesystems=flare:home
#PBS -l select=256

# Get the directory of this script
# whereami="$(dirname "$(realpath "$0")")"
# cd "${whereami}" || {
# 	echo "Failed to change directory to ${whereami}"
# 	exit 1
# }

setup_env() {
  cd /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/large-batch-training/tok50M-n512/Megatron-DeepSpeed || {
    echo "Failed to change directory to /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/large-batch-training/tok50M-n512/Megatron-DeepSpeed"
    exit 1
  }
	PBS_O_WORKDIR="$(pwd)"
	export PBS_O_WORKDIR
	export http_proxy="http://proxy.alcf.anl.gov:3128"
	export https_proxy="http://proxy.alcf.anl.gov:3128"
	export ftp_proxy="http://proxy.alcf.anl.gov:3128"
	export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
	export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"

	# shellcheck disable=SC1090
  source <(curl -L https://bit.ly/ezpz-utils)
	ezpz_setup_env
	log_message INFO "Using: $(which python3)"
}

train_model() {
	MODEL_ARCH=smollm3-3B \
		OPT=ipex.fusedlamb \
		NLAYERS=12 \
		GRAD_ACC_STEPS=2 \
		MICRO_BATCH=1 \
		USE_ACTIVATION_CHECKPOINTING=0 \
		ZERO_STAGE=0 \
		LR_DECAY_STYLE=constant \
		TOKENIZER_TYPE=HFTokenizer \
		TOKENIZER_MODEL=google/gemma-7b \
		DATA_FILE_LIST=ALCF/data-lists/aurora/olmo-mix-1124.txt \
		bash "${PBS_O_WORKDIR}/train_alcf.sh" \
		"$@"
}

main() {
	setup_env
	train_model "$@"
}

main "$@"

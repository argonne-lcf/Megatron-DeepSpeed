#!/bin/bash --login
#PBS -q prod
#PBS -A AuroraGPT
#PBS -j oe
#PBS -l walltime=06:00:00,filesystems=flare:home
#PBS -l select=256

setup_env() {
	cd "${PBS_O_WORKDIR}" || {
		echo "Failed to change directory to ${PBS_O_WORKDIR}"
		exit 1
	}
	# shellcheck disable=SC1090
	source <(curl -L https://bit.ly/ezpz-utils)
	ezpz_setup_env
	log_message INFO "Using: $(which python3)"
}

#   7,064,155,541,716  [7.064 T][@ end of stage2]
# +   706,610,881,663  [0.706 T]
# -------------------------------------------------
#   7,770,766,423,379  [7.770 T][@ end of stage4]

train_model() {
	MODEL_ARCH=AuroraGPT-2B \
		TRAIN_TOKENS=7770766423379 \
		OPT=sophiag \
		LR=2.17e-5 \
		GRAD_ACC_STEPS=2 \
		MICRO_BATCH=1 \
		USE_ACTIVATION_CHECKPOINTING=0 \
		ZERO_STAGE=0 \
		LR_DECAY_STYLE=constant \
		TOKENIZER_TYPE=HFTokenizer \
		TOKENIZER_MODEL=google/gemma-7b \
		DATA_FILE_LIST=ALCF/data-lists/aurora/nvidia-math1-code2.txt \
		bash "${PBS_O_WORKDIR}/train_alcf.sh" \
		"$@"
}

main() {
	setup_env
	train_model "$@"
}

main "$@"

#!/bin/bash --login
#PBS -q prod
#PBS -A AuroraGPT
#PBS -j oe
#PBS -l walltime=06:00:00,filesystems=flare:home
#PBS -l select=256


# Optimizer Plans
#
# 1. Ongoing:
#    - Lamb
#      - 256 Nodes (50M Tok / batch, LR=2e-4)
#      - 512 Nodes (100M Tok / batch, LR=2e-4)
# 1. Up-next:
#    256 Nodes @ 50M Tok / batch  
#    - Sophia:
#      - LR=2.17e-5
#    - Muon
#      - LR=1.82e-5
#    - Muon-clip
#      - LR=2.28e-5


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

train_model() {
  MODEL_ARCH=AuroraGPT-2B \
    OPT=muonclip \
    LR=2.28e-5 \
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

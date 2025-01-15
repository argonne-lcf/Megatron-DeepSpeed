PBS_O_WORKDIR=$(pwd)
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh) && ezpz_setup_env

DS_CONFIG=./ALCF/examples/finetune_llama3/ds_config.json
DS_CONFIG_EMPTY=./ALCF/examples/finetune_llama3/ds_config_empty.json
DATASET_PATH="./dataset/alpaca_data.json"
# DATASET_PATH=./ALCF/examples/finetune_llama3/alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
# HF_LLAMA_PATH=/flare/Aurora_deployment/meta-llama/70B/Llama-3.3-70B-Instruct
# HF_LLAMA_PATH=/home/foremans/.llama/checkpoints/Llama3.3-70B-Instruct
# HF_LLAMA_PATH=./Llama-2-7B-hf
# head -n 1 "${PBS_NODEFILE}" > nodefile-0
#ezpz_setup_job nodefile-0
# HF_LLAMA_PATH=/data/llama-2-7b-hf/
# weights link: https://huggingface.co/huggyllama/llama-7b
#

MODEL_NAME="${MODEL_NAME:-"Llama-3.3-70B-Instruct"}"

machine_name=$(ezpz_get_machine_name)
if [[ "${machine_name}" == "aurora" ]]; then
  BACKEND="ccl"
  HF_LLAMA_PATH="${MODEL_NAME}"
  # HF_LLAMA_PATH=/flare/Aurora_deployment/meta-llama/70B/Llama-3.3-70B-Instruct
elif [[ "${machine_name}" == "polaris" ]]; then
  BACKEND="nccl"
  # HF_LLAMA_PATH="Llama-3.2-1B"
  # HF_LLAMA_PATH=Llama-3.3-70B-Instruct
fi

HF_LLAMA_PATH="${MODEL_NAME}"
HF_CONFIG="${HF_LLAMA_PATH}/config.json"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
ZERO_STAGE="${ZERO_STAGE:-0}"
TP="${TP:-1}"
PP="${PP:-1}"
WORLD_SIZE="${WORLD_SIZE:-${NGPUS}}"
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * WORLD_SIZE / (TP * PP)))
# require to align with weight dimensions
# HIDDEN_SIZE=4096
# FFN_HIDDEN_SIZE=11008
# NUM_LAYERS=32
# NUM_HEADS=32
echo "Model: ${MODEL_NAME}"
echo "HF_CONFIG: ${HF_CONFIG}"
cat "${HF_CONFIG}" | jq .
HIDDEN_SIZE=$(cat "${HF_CONFIG}" | jq -r '.hidden_size')
FFN_HIDDEN_SIZE=$(cat "${HF_CONFIG}" | jq -r '.intermediate_size')
NUM_LAYERS=$(cat "${HF_CONFIG}" | jq -r '.num_hidden_layers')
NUM_HEADS=$(cat "${HF_CONFIG}" | jq -r '.num_attention_heads')
NUM_KV_HEADS=$(cat "${HF_CONFIG}" | jq -r '.num_key_value_heads')
MAX_SEQ_LENGTH=$(cat "${HF_CONFIG}" | jq -r '.max_position_embeddings')
SEQ_LENGTH=512
######################################

printf "GLOBAL_BATCH_SIZE: %s\n" $GLOBAL_BATCH_SIZE
printf "MICRO_BATCH_SIZE: %s\n" $MICRO_BATCH_SIZE
printf "ZERO_STAGE: %s\n" $ZERO_STAGE
printf "TP: %s\n" $TP
printf "PP: %s\n" $PP
printf "WORLD_SIZE: %s\n" $WORLD_SIZE
printf "HIDDEN_SIZE: %s\n" $HIDDEN_SIZE
printf "FFN_HIDDEN_SIZE: %s\n" $FFN_HIDDEN_SIZE
printf "NUM_LAYERS: %s\n" $NUM_LAYERS
printf "NUM_HEADS: %s\n" $NUM_HEADS
printf "NUM_KV_HEADS: %s\n" $NUM_KV_HEADS
printf "SEQ_LENGTH: %s\n" $SEQ_LENGTH

MEGA_DS_LLAMA_PATH="converted_hf_ckpts/${MODEL_NAME}-MDS-GBS${GLOBAL_BATCH_SIZE}-ZS${ZERO_STAGE}-TP${TP}-PP${PP}"
mkdir -p $(dirname $MEGA_DS_LLAMA_PATH)

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################
cat <<EOT >$DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "gradient_accumulation_steps": 1,
  "optimizer": {
      "type": "Adam",
      "params": {
          "lr": 1e-4
      }
  },
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

cat <<EOT >$DS_CONFIG_EMPTY
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "gradient_accumulation_steps": 1,
  "optimizer": {
      "type": "Adam",
      "params": {
          "lr": 1e-4
      }
  },
  "zero_optimization": {
    "stage": $ZERO_STAGE 
  },
  "bf16": {
    "enabled": true
  }
}
EOT

if [ "$1" = "convert_hf2mds" ]; then
  DS_CONFIG_PATH="./ALCF/examples/finetune_llama3/ds_config_empty.json"
elif [ "$1" = "convert_mds2hf" ]; then
  DS_CONFIG_PATH="./ALCF/examples/finetune_llama3/ds_config_empty.json"
else
  DS_CONFIG_PATH="./ALCF/examples/finetune_llama3/ds_config.json"
fi

# --hf-ckpt-num-shards 2 \
covert_hf2mds_args="launch python3 tools/hf2megads_weight_converter.py \
--hf-ckpt-dir ${HF_LLAMA_PATH} \
--load-mode auto \
--save ${MEGA_DS_LLAMA_PATH}"

# --hf-ckpt-num-shards 2 \
covert_mds2hf_args="launch python3 tools/hf2megads_weight_converter.py \
--hf-ckpt-dir ${HF_LLAMA_PATH} \
--load-mode auto \
--to-hf-ckpt \
--load ${MEGA_DS_LLAMA_PATH} \
--save ${HF_LLAMA_PATH}'-hf-out' "

finetune_args="launch python3 finetune_llama.py \
--load ${MEGA_DS_LLAMA_PATH}"

comm_args+=(
  "--tensor-model-parallel-size=${TP}"
  "--pipeline-model-parallel-size=${PP}"
  "--lr-warmup-iters=2000"
  "--weight-decay=0.1"
  "--clip-grad=1"
  "--num-layers=${NUM_LAYERS}"
  "--hidden-size=${HIDDEN_SIZE}"
  "--num-attention-heads=${NUM_HEADS}"
  "--finetune"
  "--ffn-hidden-size=${FFN_HIDDEN_SIZE}"
  "--num-key-value-heads=${NUM_KV_HEADS}"
  "--attention-dropout=0"
  "--hidden-dropout=0"
  "--no-query-key-layer-scaling"
  "--disable-bias-linear"
  "--normalization=rmsnorm"
  "--use-rotary-position-embeddings"
  "--untie-embeddings-and-output-weights"
  "--swiglu"
  "--seq-length=${SEQ_LENGTH}"
  "--max-position-embeddings=${MAX_SEQ_LENGTH}"
  "--micro-batch-size=${MICRO_BATCH_SIZE}"
  "--global-batch-size=${GLOBAL_BATCH_SIZE}"
  "--train-iters=3500"
  "--lr=2e-5"
  "--tensorboard-dir=tensorboard_output"
  "--lr-decay-iters=320000"
  "--lr-decay-style=cosine"
  "--log-interval=1"
  "--eval-iters=100"
  "--eval-interval=100"
  "--data-path=${DATASET_PATH}"
  "--save-interval=1500"
  "--split=100,0,0"
  "--bf16"
  "--zero-stage=${ZERO_STAGE}"
  "--tokenizer-type=HFTokenizer"
  "--tokenizer-model=meta-llama/${MODEL_NAME}"
  "--deepspeed_config=${DS_CONFIG_PATH}"
  "--deepspeed"
  "--distributed-backend=$BACKEND"
  "--num-workers=0"
  "--no-masked-softmax-fusion"
  "--no-bias-gelu-fusion"
  "--no-bias-dropout-fusion"
  "--no-gradient-accumulation-fusion"
  "--repeated-dataloader"
)

# "--optimizer=adamw"
# --tokenizer-model meta-llama/Llama-2-7B-hf \
# --tokenizer-type HFTokenizer \
# --tokenizer-model 'file:///flare/Aurora_deployment/meta-llama/70B/Llama-3.3-70B-Instruct' \
# --tokenizer-model meta-llama/Llama-3-70B-Instruct \
# --tokenizer-model ${HOME}/.llama/checkpoints/Llama3.3-70B-Instruct \
#
if [ "$1" = "convert_hf2mds" ]; then
  task_args="$covert_hf2mds_args"
elif [ "$1" = "convert_mds2hf" ]; then
  task_args="$covert_mds2hf_args"
else
  task_args="$finetune_args"
fi

full_cmd="$task_args ${comm_args[*]}"

OUTFILE="finetune-llama-$(tstamp).log"
printf "full_cmd: %s\n" "${full_cmd}" | tee -a "${OUTFILE}"
eval "$full_cmd" | tee -a "${OUTFILE}"
wait $!

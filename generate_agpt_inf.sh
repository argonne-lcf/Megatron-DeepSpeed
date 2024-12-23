#!/bin/bash
CHECKPOINT_PATH=/flare/Aurora_deployment/AuroraGPT-7B/Megatron-DeepSpeed/checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/global_step37240

TM=/flare/Aurora_deployment/AuroraGPT/datasets/dolma/utils/tokenizer.model
b=1
mp=1
experts=1
nodes=1
gpus=1
use_tutel=""
ds_inference=""
#ds_inference="--ds-inference"

export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ROOT="/flare/Aurora_deployment/intel/ccl/_install_release_2021_13"
export LD_LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LD_LIBRARY_PATH
export CPATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/include:$CPATH
export LIBRARY_PATH=/flare/Aurora_deployment/intel/ccl/_install_release_2021_13/lib:$LIBRARY_PATH
launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
#launch_cmd="python "
L=32
H=4096
A=32
FH=11008
#experts1=${experts[$k]}
#--ds-inference \
program_cmd="run_megatron.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --ffn-hidden-size $FH \
       --num-attention-heads $A \
       --max-position-embeddings 4096 \
       --tokenizer-type Llama2Tokenizer \
       --bf16 \
       --deepspeed \
       --deepspeed_config ./ALCF/ds_config_agpt_inference.json \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 4096 \
       --out-seq-length 4096 \
       --temperature 1.0 \
       --tokenizer-model $TM \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 0 \
       --no-gradient-accumulation-fusion \
       --no-async-tensor-model-parallel-allreduce \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --use-checkpoint-opt_param-scheduler \
       --lr 0.0003 \
       --finetune \
       --load $CHECKPOINT_PATH \
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd

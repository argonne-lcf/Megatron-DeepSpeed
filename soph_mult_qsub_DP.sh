## Author: Eugene Ku
## Issue: Oddly, Sophia uses more memory than polaris for some reason.
## Extra Libs (Libraries): 
## 1. Will need Apex
## a. download apex b. cd apex; python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
## 2. pip install deepspeed --upgrade (I used 0.15.1)

export PBS_O_WORKDIR=$(dirname $0 | xargs realpath)
cd $PBS_O_WORKDIR

## SCALE CONFIGS
export TRAIN_ITER=5000
export SP=1
MICRO_BATCH=1
export MICRO_BATCH=$(($SP * $MICRO_BATCH)) ## Single copy of model batch size
export DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt
# export NLAYERS=2
export NLAYERS=10
export OPT=adamw
export SAVE_INTERVAL=100
export GRAD_ACC_STEPS=1
export NO_FLASH_ATTN=1
export SOPHIA=1 ## Sophia

## MODEL ARGUEMNTS
if [ -n $SOPHIA ]; then
    . venvs/2024-08-08/bin/activate
fi

bash $PBS_O_WORKDIR/train_llama_alcf.sh


# QUEUE=by-node
# qsub -V -A datascience -q $QUEUE -l select=2 -l walltime=16:00:00,filesystems=eagle:home $PBS_O_WORKDIR/train_llama_alcf.sh

# QUEUE=debug-scaling
# qsub -V -A datascience -q $QUEUE -l select=4 -l walltime=1:00:00,filesystems=eagle:home $PBS_O_WORKDIR/train_llama_alcf.sh

# echo Submitted a job with PBS_DIR at $PBS_O_WORKDIR on $QUEUE queue.
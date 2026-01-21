#!/bin/bash
#PBS -q workq
#PBS -A Aurora_deployment
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=12
#PBS -l filesystems=gila:home
source /gila/Aurora_deployment/AuroraGPT/soft/conda.sh

cd ${PBS_O_WORKDIR}
export INPUT_DIR=/gila/Aurora_deployment/AuroraGPT/datasets/dolma/data_v1.7_Llama2Tokenizer
python /gila/Aurora_deployment/AuroraGPT/soft/tokenization/get_meta_data.py $INPUT_DIR --output dolma_v1.7.json

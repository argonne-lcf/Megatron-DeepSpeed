#!/bin/bash
#PBS -q workq
#PBS -A Aurora_deployment
#PBS -l walltime=1:00:00
#PBS -l nodes=16:ppn=12
#PBS -l filesystems=gila:home
cd ${PBS_O_WORKDIR}
export INPUT_DIR=/gila/Aurora_deployment/AuroraGPT/datasets/dolma/data_v1.7
export OUTPUT_DIR=${INPUT_DIR}_Llama2Tokenizer
export PPN=12
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
NUM_WORKERS=16 aprun -n $((PBS_JOBSIZE*PPN)) -N $PPN --cc depth -d 16 /gila/Aurora_deployment/AuroraGPT/soft/tokenization/tokenization.sh

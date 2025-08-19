#!/bin/bash
source /gila/Aurora_deployment/AuroraGPT/soft/conda.sh >& /dev/null
source /gila/Aurora_deployment/AuroraGPT/soft/local_rank.sh
export INPUT_DIR=${INPUT_DIR:-"/gila/Aurora_deployment/AuroraGPT/datasets/dolma/data"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/gila/Aurora_deployment/AuroraGPT/datasets/dolma/data_Llama2Tokenizer"}
export NUM_WORKERS=${NUM_WORKERS:-32}
export TOKENIZER_TYPE=${TOKENIZER_TYPE:-"Llama2Tokenizer"}
export input_dir=${INPUT_DIR}
export array=($(find $input_dir -type f -name "*.json.gz"))
export nfiles=${#array[@]}

if [ $RANK -eq 0 ]; then
    echo "Input folder: $input_dir"
    echo "Number of files: $nfiles"
    echo "Output folder: $OUTPUT_DIR"
    echo "num_workers: ${NUM_WORKERS}"
fi
for (( i=$RANK; i<$nfiles; i+=$WORLD_SIZE ))
do
    input_json=${array[i]}
    output_json=${input_json%%.json*}
    o_dir=$(dirname $output_json)
    o_base=$(basename $output_json)
    o_dir="${OUTPUT_DIR}/${o_dir##${INPUT_DIR}}"
    [ -e $o_dir ] || mkdir -p $o_dir
    output_json=$o_dir/$o_base
    echo "$RANK: $output_json"
    if [ -e ${output_json}_text_document.idx ]; then
	echo "$RANK: $input_json is already tokenized"
    else
	python /gila/Aurora_deployment/AuroraGPT/soft/tokenization/preprocess_data.py --input $input_json --output-prefix $output_json --tokenizer-type Llama2Tokenizer --tokenizer-model /gila/Aurora_deployment/AuroraGPT/soft/tokenization/tokenizer.model --workers $NUM_WORKERS
    fi
done

This folder contains scripts to convert the dolma datasets into megatron-deepspeed form.

Tokenizing the dataset

```bash
#!/bin/bash
#PBS -l select=4
#PBS -A Aurora_deployment
#PBS -l walltime=1:00:00
#PBS -q debug
INPUT_DIR=INPUT_TO_DIRECTORY_OF_JSONS OUTPUT_DIR=OUTPUT_DIRECTORY aprun -n 48 -N 12 --cc depth -d 16 ./tokenization.sh
```


After the tokenization, we suggest you to validate it with
```
get_meta_data.py OUTPUT_DIRECTORY --output dolma_v1.7.json
```
in which OUTPUT_DIRECTORY is the folder the tokenized data located.


We can then generate the file list
```
python3 gen_file_list.py dolma_v1.7.json 
```
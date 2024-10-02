## Latest News
* [9/12/2024] Adjusted ALCF/helpers.sh setOutput to accept an input checkpoint. setOutput() sets $CKPT_DIR which is both the input and output dir.
* [9/12/2024] Adjusted train_aGPT_7B.sh (and made a chia version) to take in the input for setOutput() and specify a d$DATA_FILE_LIST before calling setup, which makes the run_cmd - need to test with my own inputs
* [9/12/2024] My prior venv environment broke. Sam helped me get the environment to get this running again. Note: Do not share your venv for this repo with other people.
* [9/13/2024] Changed DEVICE=get_torch_device_type() to get_torch_device() in ALCF/training.sh and pretrain_gpt_alcf.py
* [9/24/2024] Have steps 1, 2, & 3 working. Sam helping me struggle through Step 4 on Aurora (down tomorrow, also down long term after this month). When moving files and file list over, note that this take significant time. Note also that directories in the data_list.txt file (weighting file) need to be updated to where the data is on whatever system you moved to (sed is your friend). I have an env_info.sh file I use to load my environment on Aurora in an interactive node. Successfully running on two nodes, world size of 24, with a micro-batch size of 1 (set in helpers.sh).
* [9/25/2024] Seem to have gotten training working, but it is not continuing from the checkpoint. Possible reasons are that the checkpoint is in the wrong format. There is an inspect_checkpoint.py and an adjust_checkpoint.py in the mds_checkpoint folder (for now, will think about where to move it later). Still getting some errors i.e., "[rank110]: RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding: [rank110]: 	size mismatch for weight: copying a param with shape torch.Size([32000, 2048]) from checkpoint, the shape in current model is torch.Size([32000, 4096])."
* [9/27/2024] Added some changes to flags for training in helpers.sh, especially related to --no-load-optim, --no-load-rng, and getting rid of --use-checkpoint-opt_params_scheduler which seem to load checkpoints. I also added --fine-tune because Azton had it in his and it seems to work for him. We have the mystery of why my MDS wants a different directory structure for model weights than his does.

## Key Scripts

| Order | Task                 | Scripts                     | Dependencies edited / Notes |
| :---: | :---:                | :---:                       | :---: |
| 1     | Data Tokenization    | tokenize repo               | [working] via preprocess_data.py found in dolma data directory on Polaris |
| 2     | Data Weights File    | ALCF/utils/gen_file_list.py | [dependencies working] megatron/data/merge_datasets_split.py, megatron/data/merge_dataset_qsub.sh, ALCF/utils/get_meta_data.sh&py, ALCF/utils/gen_file_list.py |
| 3     | HF to MDS Conversion | tools-AZTON repo            | [untested] Need to test on TinyLlama |
| 4     | Continue Training    | train_llama_chia_qsub.sh    | [working] train_aGPT_7B_chia.sh, ALCF/helpers.sh, megatron/training.py, pretrain_aGPT_alcf.py |

## Current To-Do

1. Train on 100% SlimPajama and examine loss curve
2. Mix in other data and examine loss curve

## Step 1

Go to tokenize repo and follow that. For the very last step of merging dataset into fewer files, go to Megatron-DeepSpeed-SAM/megatron/data directory and use merge_dataset_qsub.sh. You will need to specify all directories that contain files related to the particular group you are merging and how many samples to include per file. Note that for your prefixes you are going to have to include numbers after a '-'. This is the specific character that will be recognized by the code finding prefixes later.

## Step 2

Now that you have your merged files of about 5GB each (after having identified the right number of samples) you are ready to go to this repo under ALCF/utils to run the get_meta_data.sh file which will give you a json. You will take that json and feed it into gen_file_list.py. Note that in order for this to work, you will have to go into the code and manually specify epochs appropriately.

Helpful one liners if you mess up file prefixes and need to make it so that it will work with the code:

Changing a _ into a -. This only works well if you only have one _ in your prefix name:

```
rename '_' '-' -- *_*
```

Changing the final extension from .bin.idx to .idx, in case you messed up somewhere in coding file names:

```
rename .bin.idx .idx *.bin.idx
```

## Step 3

This step can actually happen at any time before Step 4. Follow instructions in tools-AZTON repo.

## Step 4

You are back in this repo and this time you are running something like train_llama_chia_qsub.sh. Main trick is getting environment set up. Some notes for Sunspot.

Get your keys right and add the following to ~/.ssh/config:

```
Host github.com
     User git
     hostname ssh.github.com

Host github.com 
     Port 443
     ProxyCommand /usr/bin/socat - PROXY:proxy.alcf.anl.gov:%h:%p,proxyport=3128
```

Then you can start the install of things:

```
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128

module use /soft/modulefiles
module load frameworks/2024.2.1_u1
module load mpich

#Navigate to project directory
cd [project-dir="/eagle/argonne_tpc_CNDA"]
git clone git@github.com:BriarRose-ANL/Megatron-DeepSpeed-SAM.git

cd Megatron-DeepSpeed-SAM
git clone https://github.com/saforem2/ezpz deps/ezpz
export PBS_O_WORKDIR=$(pwd) && source deps/ezpz/src/ezpz/bin/utils.sh && ezpz_setup_python && ezpz_setup_job
export PYTHONUSERBASE="/home/chian/candle_aesp_chia/Megatron-DeepSpeed-SAM/venvs/aurora_nre_models_frameworks-2024.2.1_u1"
python3 -m pip install -e deps/ezpz --require-virtualenv
```

# Setup
Sam tip for mpi4py build issue:

```
mv /lus/eagle/projects/argonne_tpc/chia-llama2/venvs /lus/eagle/projects/argonne_tpc/chia-llama2/venvs.bak
git clone https://github.com/saforem2/ezpz deps/ezpz
export PBS_O_WORKDIR=$(pwd) && source deps/ezpz/src/ezpz/bin/utils.sh && ezpz_setup_python && ezpz_setup_job
which python
python3 -c 'import mpi4py'
```
then summary of https://gist.github.com/saforem2/2f2549894d9c65ed2edcfe6b1dbe6a70

```
# clone repo + navigate into it
git clone git@github.com:BriarRose-ANL/Megatron-DeepSpeed-SAM.git
cd Megatron-DeepSpeed-SAM
#put in from above
python3 -m pip install -e deps/ezpz --require-virtualenv
# deepspeed
git clone https://github.com/microsoft/DeepSpeed deps/DeepSpeed
cd deps/DeepSpeed && bash install.sh |& tee install.log && cd -
# upgrade W&B
python3 -m pip install --upgrade wandb
# launch:
PBS_O_WORKDIR=$(pwd) bash train_llama_chia_qsub.sh
```

things being tried on Sunspot when you can't install DeepSpeed because XGBoost is missing:

```
python3 -m pip install xgboost "numpy<2"
python3 -m pip install -e deps/DeepSpeed
python3 -m pip install deepspeed
```


# Usage

Some of the below is guessed out.

Step 1: Data Tokenization: There is reference to a preprocess_data.py script in /gila/Aurora_deployment in Sunspot. This is the script we would need. I found something named the same in /lus/eagle/projects/datasets/dolma/utils. Given what I know about the dolma folder being a seemingly older version of what Huihao has (my guess), this might also be about of data, but I'm going to try it.

Step 2: Data Weights File: Tried to match up Huihao's script with something existing in /lus/eagle/projects/datasets/dolma, but the gen_file_list.py script there is much shorter than Huihao's script, so failing to file a similar workspace on Polaris that is as up-to-date as Huihao's version.

# Training
## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers 5
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

For T5 use the same preprocessing as BERT, perhaps renaming it to:
<pre>
       --output-prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --workers 5
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

## BERT Pretraining


The [`examples/pretrain_bert.sh`](./examples/pretrain_bert.sh) script runs single GPU 345M parameter BERT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--lr-warmup-fraction`. While this is single GPU training, the batch size specified by `--micro-batch-size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global-batch-size` which is the batch size per iteration. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train-iters` as the training iterations requested. Alternatively, one can provide `--train-samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr-decay-iters`, one will need to provide `--lr-decay-samples`.

The logging, checkpoint-saving, and evaluation intervals are specified. Checkpointing the activations facilitates the training of larger models and/or batches. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

To run `examples/pretrain_bert.sh`, make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `VOCAB_FILE`, and `DATA_PATH`. Make sure to set these variables to their paths in the container. Then launch the container with Megatron and necessary paths mounted (as explained in [Setup](#setup)) and run the example script.

## GPT Pretraining

The `examples/pretrain_gpt.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

`examples/pretrain_gpt.sh` can be launched the same way as described for BERT. Set the env vars and make any other modifications, launch the container with appropriate mounts, and run the script.

## T5 Pretraining

Very similar to BERT and GPT, the `examples/pretrain_t5.sh` script runs single GPU "base" (~220M parameter) T5 pretraining. The primary difference from BERT and GPT is the addition of the following arguments to accommodate the T5 architecture:

* `--kv-channels` sets the inner dimension of the "key" and "value" matrices of all attention mechanisms in the model. For BERT and GPT this defaults to the hidden size divided by the number of attention heads, but can be configured for T5.

* `--ffn-hidden-size` sets the hidden size in the feed-forward networks within a transformer layer. For BERT and GPT this defaults to 4 times the transformer hidden size, but can be configured for T5.

* `--encoder-seq-length` and `--decoder-seq-length` set the sequence length for the encoder and decoder separately.

All of the other arguments remain as they were for BERT and GPT pretraining. Run this example with the same steps described above for the other scripts.

## Distributed Pretraining

The `examples/pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables. See the official PyTorch [documentation](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the `torchrun` elastic launcher (equivalent to `python -m torch.distributed.run`) are the only additional requirements to adopt distributed training. See any of `examples/pretrain_{bert,gpt,t5}_distributed.sh` for more details.

We use two types of parallelism: data and model parallelism. We facilitate two distributed data parallel implementations: a simple one of our own that performs gradient all-reduce at the end of back propagation step, and Torch's distributed data parallel wrapper that overlaps gradient reduction with back propagation computation. To switch between these two options use `--DDP-impl local` or `--DDP-impl torch`, respectively. As expected, Torch distributed data parallelism is more efficient at larger model sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 76% when Torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use tensor model parallelism (splitting execution of a single transformer module over multiple GPUs, see Section 3 of [our paper](https://arxiv.org/pdf/1909.08053.pdf)), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use sequence parallelism specify `--sequence-parallel`, which requires tensor model parallel as it split among the same GPUs (more details in Section 4.2.2 of [our paper](https://arxiv.org/pdf/2205.05198.pdf)).

To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches, see Section 2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

<!-- The number of microbatches in a per-pipeline minibatch is controlled by the `--num-microbatches-in-minibatch` argument. With `WORLD_SIZE` GPUs, `TENSOR_MP_SIZE` tensor-model-parallel size, `PIPELINE_MP_SIZE` pipeline-model-parallel-size, `WORLD_SIZE`/(`TENSOR_MP_SIZE` * `PIPELINE_MP_SIZE`) GPUs will be used for data parallelism. The default values for `--tensor-model-parallel-size` and `--pipeline-model-parallel-size` is 1, which will not implement either form of model parallelism. -->

We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`:

Other than these minor changes, the distributed training is identical to the training on a single GPU.

The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num-layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).

## Activation Checkpointing and Recomputation

To reduce GPU memory usage so deploy a large model to a training system, we support activation checkpointing and recomputation. We support two levels of recompute granularity: `selective` and `full`. Selective recomputation is the default and recommended in almost all cases. It saves the activations that take less space and are expensive to recompute and recomputes activations that take a lot of space but are relatively cheap to recompute (see [our paper](https://arxiv.org/pdf/2205.05198) for details). To enable selective activation recompute simply use `--recompute-activations`.

For cases where memory is very tight, `full` checkpointing saves just the inputs to a transformer layer, or a block of transformer layers, and recomputes everything else. To turn on full activation recompute use `--recompute-granularity full`. When using full activation recomputation, there are two methods: `uniform` and `block`, chosen using the `--recompute-method` argument.

* Uniform method uniformly divides the Transformer layers into groups of layers and stores the input activations of each group in the memory. The baseline group size is 1 and, in this case, the input activation of each Transformer layer is checkpointed. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage thus enables running a bigger model. For example, when using the number of layers per group of 4, the input activation of each group of 4 Transformer layers is checkpointed.

* Block method checkpoints the input activations of a set number of individual Transformer layers per pipeline stage and do the rest of layers without any checkpointing. This method can be used to skip checkpointing some Transformer layers until the GPU memory is fully used, which is applicable only when there is unused GPU memory. Checkpointing fewer transformer layers avoids unnecessary activation recomputation in the backprop thus improves training performance. For example, when we specify 5 layers to checkpoint of 8 layers per pipeline stage, the input activations of only the first 5 Transformer layers are checkpointed and activation recomputation for the rest 3 layers is not needed in the backprop.


## Distributed Optimizer

Usage: `--use-distributed-optimizer`. Compatible with all model and data types.

The distributed optimizer is a memory savings technique, whereby the optimizer state is evenly distributed across data parallel ranks (versus the traditional method of replicating the optimizer state across data parallel ranks). As described in [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), our implementation distributes all optimizer state that does not overlap with the model state. For example, when using fp16 model params, the distributed optimizer maintains its own separate copy of fp32 main params & grads, which are distributed across DP ranks. When using bf16 model params, however, the distributed optimizer's fp32 main grads are the same as the model's fp32 grads, and so the grads in this case are not distributed (although the fp32 main params are still distributed, as they are separate from the bf16 model params).

Theoretical memory savings vary depending on the combination of the model's param dtype and grad dtype. In our implementation, the theoretical number of bytes per parameter is (where 'd' is the data parallel size):

| | Non-distributed optim | Distributed optim |
|-|-|-|
| fp16 param, fp16 grads | 20 | 4 + 16/d |
| bf16 param, fp32 grads | 18 | 6 + 12/d |
| fp32 param, fp32 grads | 16 | 8 + 8/d |

## FlashAttention

Usage: `--use-flash-attn`. Support attention head dimensions at most 128.

[FlashAttention](https://github.com/HazyResearch/flash-attention) is a fast and
memory-efficient algorithm to compute exact attention. It speeds up model
training and reduces memory requirement.

To install FlashAttention:
```sh
pip install flash-attn
```

## GPT-3 Example

In `examples/pretrain_gpt3_175B.sh` we have provided an example of how to configure Megatron to run [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way and 16-way tensor and pipeline parallelism, respectively. With options `global-batch-size 1536` and `rampup-batch-size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incrmeental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.


## Retro

See:

- `tools/retro/README.md` for an overview.
- `tools/retro/examples/get_preprocess_cmd.sh` for an example of common preprocessing arguments.
- `tools/retro/examples/preprocess_data.sh` for an example of how to preprocess data.
- `tools/retro/examples/pretrain_model.sh` for an example of how to pretrain a model.

Retro is a retrieval-enhanced model that is based on GPT. As described in [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426), Retro retrieves from a database of document chunks by performing locality search using a sample's tokens. The retrieval database can be large -- often billions or even trillions of tokens -- and provides a more efficient storage mechanism of factual knowledge, when compared to storing factual knowledge implicitly within the network's parameters.

Using Retro requires two steps: 1) preprocessing the retrieval database and pretraining neighbors, and 2) pretraining a model using this data. Please see `tools/retro/README.md` for a detailed overview.

<!--
## REALM Pipeline
We are working on implementing the [REALM](https://arxiv.org/pdf/2002.08909.pdf) system. The following sections (will) reflect the three stages of training it. For now it's just the ICT code.
Loosely, they are pretraining the retriever modules, then jointly training the language model and the retriever, and then finetuning a question answering head on the language model with fixed retriever.

### Inverse Cloze Task (ICT) Pretraining
1. Have a corpus in loose JSON format with the intention of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document.
Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. For the original REALM system, we construct two datasets, one with the title of every document, and another with the body.
Refer to the following script
<pre>
python preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 5  # works well for 10 CPU cores. Scale up accordingly.
</pre>

2. Use a custom samples mapping function in place of `megatron/data/realm_dataset_utils.get_block_samples_mapping` if required. To do this, you will need to implement a new function in C++ inside of `megatron/data/helpers.cpp`. The samples mapping data structure is used to select the data that will constitute every training sample in advance of the training loop.
 The samples mapping is responsible for holding all of the required metadata needed to construct the sample from one or more indexed datasets. In REALM, the samples mapping contains the start and end sentence indices, as well as the document index (to find the correct title for a body) and a unique ID for every block.
3. Pretrain a BERT language model using `pretrain_bert.py`, with the sequence length equal to the block size in token ids. This model should be trained on the same indexed dataset that is used to supply the blocks for the information retrieval task.
In REALM, this is an uncased bert base model trained with the standard hyperparameters.
4. Use `pretrain_ict.py` to train an `ICTBertModel` which uses two BERT-based encoders to encode queries and blocks to perform retrieval with.
The script below trains the ICT model from REALM. It refrences a pretrained BERT model (step 3) in the `--bert-load` argument. The batch size used in the paper is 4096, so this would need to be run with data parallel world size 32.
<pre>
python pretrain_ict.py \
    --num-layers 12 \
    --num-attention-heads 12 \
    --hidden-size 768 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-head-size 128 \
    --train-iters 100000 \
    --activations-checkpoint-method uniform \
    --bert-load /path/to/pretrained_bert \
    --load checkpoints \
    --save checkpoints \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --vocab-file /path/to/vocab.txt \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --save-interval 3000 \
    --query-in-block-prob 0.1 \
    --fp16

</pre>

### Building an Index of Block Embeddings
After having trained an ICT model, you can now embed an entire dataset of blocks by creating a `BlockData` structure. After that has been saved, you can load it
and wrap it with a `FaissMIPSIndex` to do fast similarity search which is key in the learned information retrieval pipeline. The initial index can be built with the following script, meant to be run in an interactive session. It can leverage multiple GPUs on multiple nodes to index large datasets much more quickly.

<pre>
python tools/create_doc_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --ict-head-size 128 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --activations-checkpoint-method uniform \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-load /path/to/pretrained_ict \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --block-data-path embedded_blocks.pkl \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file /path/to/vocab.txt \
    --num-workers 2 \
    --fp16
</pre>

-->

# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on fewer GPUs in downstream tasks. The following script accomplishes this. This example reads in a GPT model with 4-way tensor and 4-way pipeline model parallelism and writes out a model with 2-way tensor and 2-way pipeline model parallelism.

<pre>
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir checkpoints/gpt3_tp4_pp4 \
        --save-dir checkpoints/gpt3_tp2_pp2 \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Text Generation

We have included a simple REST server to use for text generation in `tools/run_text_generation_server.py`. You run it much like you would start a pretraining job, specifying an appropriate pretrained checkpoint. There are also few optional parameters: `temperature`, `top-k`and `top-p`. See `--help` or the source file for more information. See [examples/run_text_generation_server_345M.sh](examples/run_text_generation_server_345M.sh) for an example of how to run the server.

Once the server is running you can use `tools/text_generation_cli.py` to query it, it takes one argument which is the host the server is running on.

<pre>
tools/text_generation_cli.py localhost:5000
</pre>

You can also use CURL or any other tools to query the server directly:

<pre>
curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["Hello world"], "tokens_to_generate":1}'
</pre>

See [megatron/text_generation_server.py](megatron/text_generation_server.py) for more API options.

### Detoxify GPT via Self-generation
We include an example in `examples/detxoify_lm/` to detoxify language models by leveraging the generative power of language models.

See [examples/detxoify_lm/README.md](examples/detxoify_lm/README.md) for step-by-step tutorials on how to perform domain-adaptive training and detoxify LM using self-generated corpus.


## GPT Evaluation
We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --activations-checkpoint-method uniform \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>


### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a 345M parameter model. Note that the `--strict-lambada` flag should be used to require whole word matching. Make that `lambada` is part of the file path.

<pre>
TASK="LAMBADA"

VALID_DATA=&#60;lambada path&#62;.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m
COMMON_TASK_ARGS=&#60;same as those in <a href="#wikitext-perplexity-evaluation">WikiText Perplexity Evaluation</a> above&#62;

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --activations-checkpoint-method uniform \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

Further command line arguments are described in the source file [`main.py`](./tasks/main.py)

## BERT Task Evaluation
### RACE Evaluation
The following script finetunes the BERT model for evaluation on the [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/). The `TRAIN_DATA` and `VALID_DATA` directory contain the RACE dataset as separate `.txt` files. Note that for RACE, the batch size is the number of RACE query's to evaluate. Since each RACE query has four samples, the effective batch size passed through the model will be four times the batch size specified on the command line.

<pre>
TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --activations-checkpoint-method uniform \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"

python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 3 \
       --micro-batch-size 4 \
       --lr 1.0e-5 \
       --lr-warmup-fraction 0.06
</pre>

### MNLI Evaluation
The following script finetunes the BERT model for evaluation with the [MultiNLI sentence pair corpus](https://www.nyu.edu/projects/bowman/multinli/). Because the matching tasks are quite similar, the script can be quickly tweaked to work with the [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) (QQP) dataset as well.

<pre>

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli
COMMON_TASK_ARGS=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;
COMMON_TASK_ARGS_EXT=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;

python tasks/main.py \
       --task MNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 5 \
       --micro-batch-size 8 \
       --lr 5.0e-5 \
       --lr-warmup-fraction 0.065
</pre>

# Datasets
We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, use the `--split-sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.

## Collecting GPT Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.

# Reproducibility
Megatron training is intended to be bitwise reproducible. This means that the same training config run twice in the same HW and SW environment should produce identical model checkpoints, losses and accuracy metric values (iteration time metrics may vary).

There are currently three known Megatron optimizations that break reproducibility whilst still producing almost identical training runs. They are only applicable when using NGC containers >=22.05. The following workarounds should be applied in cases where reproducibility is required:
1. When training using the `--bf16` option the backward pass of `torch.nn.functional.embedding` is non-deterministic. If reproducibility is required you should also use the option `--embedding-weights-in-fp32`. The speed and memory impact of this change is negligible.
2. Also when training using `--bf16`, reproducbility is only obtained when the checkpointing and resume schedule of training is identical. If the checkpointing schedule will change, i.e. checkpointing and resume will occur at different iterations, the option `--no-bias-gelu-fusion` should be used.
3. Flash attention is non-deterministic. If reproducibility is required do not use `--use-flash-attn`.

These sources of non-determinism are under active investigation. If you observe non-determinism in Megatron training under other circumstances please open an issue.

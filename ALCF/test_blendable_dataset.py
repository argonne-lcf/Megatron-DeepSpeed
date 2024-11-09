#!/usr/bin/env python
#
# This is to test how the blendable dataset works
#
# --------------------------------------------------

import time
import json
start_time = time.time()
from mpi4py import MPI
import os
from megatron.data.gpt_dataset import build_train_valid_test_datasets
import numpy as np
from megatron.global_vars import set_args, set_global_variables, get_args
from megatron.arguments import parse_args 
from megatron.initialize import initialize_megatron
from megatron.data.data_samplers import build_pretraining_data_loader
import datetime
import torch
from tqdm import tqdm
from megatron.core import mpu


def local_args(parser):
    group = parser.add_argument_group(title="local-dataset-args")
    group.add_argument("--num-chunks", type=int, default=1)
    group.add_argument("--start-step", type=int, default=0)
    group.add_argument("--end-step", type=int, default=-1)
    group.add_argument("--niters", type=int, default=10)
    group.add_argument("--output-folder", type=str, default="./")
    return parser

comm = MPI.COMM_WORLD
def print_rank_0(msg):
    if comm.rank==0:
        print(f" [INFO][{datetime.datetime.now()}] {msg}", flush=True)

from megatron.utils import PerfTrace, Profile

end_time = time.time()        
print_rank_0(f"Loaded python modules in {end_time - start_time} seconds")
comm.Barrier()
print_rank_0(f"Barrier synchonization time:  {time.time() - end_time} seconds")


def set_master_addr():
    import socket
    if "MASTER_ADDR" not in os.environ.keys():
        master_addr = socket.gethostname()
        comm.bcast(master_addr, root=0)
        os.environ["MASTER_ADDR"] = master_addr


def initialize():
    set_master_addr()

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"]="1"
    os.environ["MASTER_PORT"]="1234"
    args_default = {
        "hidden_size": 4096,
        "micro_batch_size": 1,
        "num_layers":1,
        "num_attention_heads":32,
        "seq_length": 2048,
        "distributed_backend": "ccl",
        "max_position_embeddings": 10000000,
        "tokenizer_type": "Llama2Tokenizer",
        "tokenizer_model": "Megatron-DeepSpeed/ALCF/tokenizer.model"
    }

    initialize_megatron(extra_args_provider=local_args,
                        allow_no_cuda=True,
                        args_defaults = args_default)
    args = get_args()
    os.makedirs(args.output_folder, exist_ok=True)
    return args
args = initialize()

if os.getenv('DLIO_PROFILER_DATASET_DIR') is not None:
    extra_trace_path = os.environ['DLIO_PROFILER_DATASET_DIR']
else:
    extra_trace_path=''
    PerfTrace.initialize_log(f"{args.trace_dir}/trace-{comm.rank}-of-{comm.size}.pfw",  f"{args.data_cache_path}:{extra_trace_path}:{args.data_path}:{args.save}:{args.load}", process_id=comm.rank)

dlp = Profile("TEST_BLENDABLEDATASET")

os.makedirs(args.trace_dir, exist_ok=True)

def get_files_corpora_weights():

    data_file_list = args.data_file_list
    print_rank_0(f"Reading data from {args.data_file_list}")
    corpora = []
    files = []
    weights = []
    with open(data_file_list, 'r') as fin:
        for f in fin.readlines():
            w, fname, c = f.split()
            weights.append(float(w))
            files.append(float(w))
            files.append(fname)
            files.append(c)
            if c not in corpora:
                corpora.append(c)
    weights = np.array(weights)/np.sum(weights)
    return files, corpora, weights
files, corpora, weights = get_files_corpora_weights()

splits_string="100,0,0"
num_samples = args.global_batch_size*args.train_iters
num_datasets = len(weights)
print_rank_0(f"Number of datasets: {num_datasets}")
print_rank_0(f"Global batch size: {args.global_batch_size}")
print_rank_0(f"Training iterations: {args.train_iters}")

train_valid_test_num_samples = [num_samples, 0, 0]
seed=args.seed
data_impl = args.data_impl
skip_warmup = not args.mmap_warmup
seq_length = args.seq_length
splits_string = "1,0,0"

# Build datasets
start_build_dataset = time.time()
print_rank_0(f"Starting to build the blendable dataset")
train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    files, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup, data_cache_path=args.data_cache_path)
end_build_dataset = time.time()

print_rank_0(f"Finished building the blendable dataset in {end_build_dataset - start_build_dataset} second")
print_rank_0(f"Total number of samples: {len(train_ds)}")
print_rank_0(f"Weights set: {weights[:min(8, num_datasets)]}")

def get_sample_info(blendable_dataset, idx):
    # corpus dataset
    cd = blendable_dataset.dataset_index[idx]
    # index within the corpus dataset
    cds = blendable_dataset.dataset_sample_index[idx]
    # dataset index within each corpus
    fcd = blendable_dataset.datasets[cd].dataset_index[cds]
    # sample index within the dataset
    fcds = blendable_dataset.datasets[cd].dataset_sample_index[cds]
    # corresponding data file
    prefix = blendable_dataset.datasets[cd].dataset_builders[fcd].prefix
    corpus = blendable_dataset.datasets[cd].dataset_builders[fcd].corpus
    #v = blendable_dataset[idx]['text']
    #norm = np.linalg.norm(v)
    return prefix, corpus, fcds

num_batches =  args.train_iters
print_rank_0(f"global_batch_size: {args.global_batch_size}")
print_rank_0(f"number of batches: {num_batches}")
print_rank_0(f"Going through all the batches")

chunks = args.num_chunks
if comm.rank == 0:
    fout = {}
    nbatches_per_chunks = num_batches//chunks
    if num_batches%chunks > 0:
        nbatches_per_chunks += 1
        
    start_batch = args.start_step
    if args.end_step == -1:
        end_batch = num_batches
    else:
        end_batch = args.end_step
    
    for i in range(start_batch//nbatches_per_chunks, end_batch//nbatches_per_chunks):
        fout[i] = open(f"samples_list_{i}-of-{chunks}.json", "w")
    
    for i in tqdm(range(start_batch, end_batch)):
        ichunk = i//nbatches_per_chunks        
        ns_corpus = {}
        for c in corpora:
            ns_corpus[c] = 0
        for j in range(args.global_batch_size):
            prefix, corpus, idx = get_sample_info(train_ds, i*args.global_batch_size+j)
            ns_corpus[corpus] +=1
            item = {
                "batch": i,
                "sample": j,
                "corpus": corpus,
                "prefix": prefix,
                "dataset_sample_index":idx
            }
            fout[ichunk].write(str(item)+"\n")
        item = {
            "batch": i,
            "histogram": ns_corpus
        }
        fout[ichunk].write(str(item) +"\n")
    for i in range(start_batch//nbatches_per_chunks, end_batch//nbatches_per_chunks):        
        fout[i].close()

        
comm.Barrier()        

# Testing data loader

start_build_dataloader = time.time()
print_rank_0(f"Starting to build the data loader")
rank_in_parallel_group = mpu.get_sequence_parallel_rank()
train_dataloader = build_pretraining_data_loader(
    train_ds, args.consumed_train_samples)
valid_dataloader = build_pretraining_data_loader(
        valid_ds, args.consumed_valid_samples)
test_dataloader = build_pretraining_data_loader(test_ds, 0)
end_build_dataloader = time.time()
print_rank_0(f"Finished building the data loader in {end_build_dataloader - start_build_dataloader} second")

print_rank_0(f"Starting loading the data")
start_loading_time = time.time()
NUM_ITEMS=1
SLEEP_TIME=10.0
@dlp.log
def compute(ct):
    time.sleep(ct)
n=0
start_time = time.time()
for i in dlp.iter(train_dataloader):
    print(f"[{comm.rank}] DATA {i}")
    n+=1
    if (n%NUM_ITEMS==0):
        print_rank_0(f"Proccessed {n}th-batch in {time.time() - start_time}")
    if n>=1000:
        break
    start_time = time.time()
end_loading_time = time.time()
print_rank_0(f"Finished loading the data ({n} batches) in {end_loading_time - start_loading_time}")

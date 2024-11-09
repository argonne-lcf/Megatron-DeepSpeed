# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import logging
import numpy as np
import torch

from deepspeed.accelerator import get_accelerator
# from megatron import print_rank_0
from megatron.core import mpu
from megatron.utils import Profile, PerfTrace
from mpi4py import MPI

from megatron.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)

dlp = Profile("DATASET")
class BlendableDataset(torch.utils.data.Dataset):
    @dlp.log
    def __init__(self, datasets, weights, size, *,
                 data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indicies.
        @dlp.log
        def _build_indices():
            start_time = time.perf_counter()
            dataset_index = np.zeros(self.size, dtype=np.int64)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from megatron.data import helpers
            helpers.build_blending_indices(dataset_index, dataset_sample_index,
                                           weights, num_datasets, self.size,
                                           torch.distributed.get_rank() == 0)
            log.info(
                "> elapsed time for building blendable dataset indices: "
                f"{time.perf_counter() - start_time:.2f} (sec)"
            )
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc
        self.dataset_index = np.zeros(self.size, dtype=np.int64)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")
            cache_hit = os.path.isfile(index_path) and os.path.isfile(sample_index_path)
            cache_success = True
            if torch.distributed.get_rank() == 0 and not cache_hit:
                print(' > WARNING: could not find index map files for blendable'
                      ' dataset, building indices on rank 0 ...', flush=True)
                dataset_index, dataset_sample_index = _build_indices()
                try:
                    log.debug(" > saving index map files")
                    start_time = time.perf_counter()
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    with open(desc_path, 'wt') as fd:
                        fd.write(desc)
                        np.save(index_path, dataset_index, allow_pickle=True)
                        np.save(sample_index_path, dataset_sample_index,
                                allow_pickle=True)
                    log.info(f" > finished saving index map files in {time.perf_counter() - start_time} seconds")
                except OSError:
                    print(f'There was an error trying to create the data cache directory ({data_cache_path})')
                    print('or a file in it. This is set with the --data-cache-path argument. Please')
                    print('ensure you have write access to this directory or specify one that you do have')
                    print('write access to.')
                    cache_success = False
                self.dataset_index = dataset_index
                self.dataset_sample_index = dataset_sample_index
            ''' I don't think the following piece of code is necessary any more; I commented them out now
            counts = get_accelerator().LongTensor([cache_success])
            torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
            if counts[0].item() != (
                    torch.distributed.get_world_size() //
                    torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()) //
                    torch.distributed.get_world_size(group=mpu.get_sequence_parallel_group())):
                log.info("Data index creation unsuccessful, exiting.")
                exit()
            '''
            torch.distributed.barrier(group=mpu.get_data_parallel_group())
            torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())
            torch.distributed.barrier(group=mpu.get_data_parallel_group())

            start_time = time.perf_counter()
            log.info(f'> loading blendable dataset index: {index_path}')
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_index.size == self.size
            log.info(f'> loading blendable dataset sample index: {sample_index_path}')
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_sample_index.size == self.size
            log.info(f'> finished loading in {time.perf_counter() - start_time} seconds')
        else:
            self.dataset_index, self.dataset_sample_index = _build_indices()


        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError('BlendedDataset size is improperly bounded')
        except IndexError:
            pass
        log.info('> size of blendable dataset: '
                     '{} samples'.format(self.size))


    def __len__(self):
        return self.size

    @dlp.log
    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx" : dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import hashlib
import os
import time

import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from megatron import print_rank_0, is_rank_0, get_args
from megatron.core import mpu
from megatron.data import helpers
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples, get_datasets_corpuses_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

from megatron.utils import PerfTrace, Profile
from mpi4py import MPI

dlp = Profile("DATASET")

@dlp.log
def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    return_doc_ids=False, *,
                                    data_cache_path=None):
    """Build train, valid, and test datasets."""

    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")

        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    data_impl, splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length, seed, skip_warmup,
                                                    data_cache_path=data_cache_path)

        # Blending dataset.
        # Parse the values.
        output = get_datasets_corpuses_weights_and_num_samples(data_prefix,
                                                      train_valid_test_num_samples)
        prefixes, corpuses, weights, datasets_train_valid_test_num_samples = output
        corpus_list = sorted(set(corpuses))
        train_num_samples, valid_num_samples, test_num_samples = map(
            sum,
            zip(*datasets_train_valid_test_num_samples)
        )

        class DatasetBuilder:
            ''' 
            This is for building individual dataset from each dataset file
            '''
            @dlp.log
            def __init__(self, prefix, corpus, data_impl, splits_string,
                         num_samples, seq_length, seed, skip_warmup,
                         return_doc_ids,
                         data_cache_path=data_cache_path, name='train'):
                self.prefix = prefix
                self.data_impl = data_impl
                self.splits_string = splits_string
                if name == 'train':
                    self.num_samples = num_samples[0]
                elif name == 'valid':
                    self.num_samples = num_samples[1]
                else:
                    self.num_samples = num_samples[2]
                self.num_samples_train_valid_test = num_samples
                self.seq_length = seq_length
                self.seed = seed
                self.skip_warmup = skip_warmup
                self.return_doc_ids = return_doc_ids
                self.data_cache_path = data_cache_path
                self.dataset = None
                self.name = name
                self.desc = prefix + f"{self.num_samples}" + f"{seq_length}" + f"{seed}"
                self.build = False
                self.corpus = corpus
            @dlp.log
            def Build(self):
                self.dataset = _build_train_valid_test_datasets_single(self.prefix, self.data_impl, self.splits_string,
                    self.num_samples_train_valid_test, self.seq_length, self.seed, self.skip_warmup, self.name, self.return_doc_ids, 
                    data_cache_path=self.data_cache_path)
                self.build = True
                return self.dataset

        class BuildConcatDataset(torch.utils.data.Dataset):
            @dlp.log
            def __init__(self, dataset_builders):
                self.dataset_builders = dataset_builders
                self.num_datasets = len(dataset_builders)
                self.num_samples = np.sum([d.num_samples for d in dataset_builders])
                self.indices=np.zeros((self.num_samples, 2), dtype=np.uint64)
                self.desc="ConcatDataset:"
                m = 0
                num_samples_list = np.array([d.num_samples for d in dataset_builders])
                self.num_samples = np.sum(num_samples_list)
                def _build_indices():
                    start_time = time.time()
                    dataset_index = np.zeros(self.num_samples, dtype=np.int64)
                    dataset_sample_index = np.zeros(self.num_samples, dtype=np.int64)
                    helpers.build_concat_indices(dataset_index, dataset_sample_index,
                                                 num_samples_list, 
                                                 self.num_datasets, 
                                                 torch.distributed.get_rank()==0)
                    print_rank_0('> elapsed time for building concat dataset indices: '
                                 '{:.2f} (sec)'.format(time.time() - start_time))
                    return dataset_index, dataset_sample_index
                
                self.dataset_index, self.dataset_sample_index = _build_indices()
                for i in range(self.num_datasets):
                    self.desc += dataset_builders[i].prefix + ","

                self.desc += f"-{self.num_samples}" + f"-{dataset_builders[0].seq_length}" + f"{dataset_builders[0].seed}"
            def __len__(self):
                return self.num_samples

            @dlp.log
            def __getitem__(self, idx):
                if idx >= self.num_samples:
                    print_rank_0(f"WARNING: index overflow encountered {idx} > {self.num_samples} for {self.dataset_builders[0].corpus}; will randomly pick one sample")
                    id = np.random.randint(self.num_samples)
                else:
                    id = idx
                i = self.dataset_index[idx]
                j = self.dataset_sample_index[idx]
                if self.dataset_builders[i].build:
                    return self.dataset_builders[i].dataset[j]
                else:
                    return self.dataset_builders[i].Build()[j]
            

        # Predetermine whether need to build the specific dataset or not. 
        start_time = time.time()
        print_rank_0(" >>> Started building datasets in distributed way ... ")

        a, b, c = [int(d) for d in splits_string.split(",")]

        train_datasets = []
        valid_datasets = []
        test_datasets = []
        # Build individual datasets.

        @dlp.log
        def build_corpus_datasets(dataset_type='train'):
            start_time = time.time()
            print_rank_0(" >>> Building {dataset_type} corpus datasets ...")
            datasets = []
            corpus_builders = {}
            corpus_weights = {}
            for c in corpus_list:
                corpus_builders[c] = []
                corpus_weights[c] = 0.0
            dataset_builders = [DatasetBuilder(prefixes[i], corpuses[i], data_impl, splits_string,
                                               datasets_train_valid_test_num_samples[i],
                                               seq_length, seed, skip_warmup,
                                               return_doc_ids,data_cache_path, dataset_type) for i in  range(len(weights))]
            for i in range(torch.distributed.get_rank(), len(weights), torch.distributed.get_world_size()):
                dataset_builders[i].Build()
            print_rank_0(f" >>> Finished building individual datasets in {time.time() - start_time} seconds")
            start_concating_time = time.time()
            for i, d in zip(range(len(weights)), dataset_builders):
                corpus_builders[d.corpus].append(d)
                corpus_weights[d.corpus] += weights[i]
            total = 0
            print_rank_0(" > number of samples for each corpus ")
            corpus_weights_achieved={}
            for c in corpus_list:
                datasets.append(BuildConcatDataset(corpus_builders[c]))
                total += datasets[-1].num_samples
                corpus_weights_achieved[c] =  float(datasets[-1].num_samples)/train_num_samples                
                print_rank_0(f"    {c}: {datasets[-1].num_samples} w={corpus_weights_achieved[c]} (expected: {corpus_weights[c]})")
            
            print_rank_0(f" > total number of samples: {total}")
            print_rank_0(f" >>> Finished concatenating datasets in {time.time() - start_concating_time} seconds")
            print_rank_0(f" >>> Finished building {dataset_type} corpus datasets in {time.time() - start_time} seconds")
            return datasets, [corpus_weights_achieved[c] for c in corpus_list]

        if a > 0:
            train_datasets, train_weights = build_corpus_datasets('train')

        if b > 0:
            valid_datasets, valid_weights = build_corpus_datasets('valid')
            
        if c > 0:            
            test_datasets, test_weights = build_corpus_datasets('test')

        # This barrier is critical to make sure that all the datasets are built once
        # and the metadata were written to the cache folder before other ranks touch them
        print_rank_0(f" >>> Rank 0 - finished building datasets in {time.time() - start_time} seconds")                
        MPI.COMM_WORLD.Barrier()
        print_rank_0(f" >>> Finished building datasets (all ranks) in distributed way in {time.time() - start_time} seconds")
        print_rank_0(f" >>> Starting to build BlendableDataset")
        # Blend.
        start_time = time.time()
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, train_weights, train_num_samples,
                                                      data_cache_path=data_cache_path)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, valid_weights, valid_num_samples,
                                                      data_cache_path=data_cache_path)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, test_weights, test_num_samples,
                                                     data_cache_path=data_cache_path)
        end_time = time.time()
        print_rank_0(f" >>> Finished building BlendableDataset in {end_time - start_time} seconds")
        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)

    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                          splits_string,
                                          train_valid_test_num_samples[0],
                                          seq_length, seed, skip_warmup,
                                          data_cache_path=data_cache_path)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                          splits_string,
                                          train_valid_test_num_samples[1],
                                          seq_length, seed, False,
                                          data_cache_path=data_cache_path)


        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                         splits_string,
                                         train_valid_test_num_samples[2],
                                         seq_length, seed, False,
                                         data_cache_path=data_cache_path)

        return (train_dataset, valid_dataset, test_dataset)

@dlp.log
def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False, *,
                                     data_cache_path=None):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = GPTDataset(name, data_prefix, documents, indexed_dataset,
                                 splits_string,
                                 train_valid_test_num_samples[index],
                                 seq_length, seed,
                                 return_doc_ids,
                                 data_cache_path=data_cache_path)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)

@dlp.log
def _build_train_valid_test_datasets_single(data_prefix, data_impl, splits_string,
                            train_valid_test_num_samples,
                            seq_length, seed, skip_warmup, name, 
                            return_doc_ids=False, *,
                            data_cache_path=None):
    """Build train, valid, and test datasets."""

    # Each rank print out information
    print_rank_0(f" >> building dataset for {data_prefix}")
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = GPTDataset(name, data_prefix, documents, indexed_dataset,
                                 splits_string,
                                 train_valid_test_num_samples[index],
                                 seq_length, seed,
                                 return_doc_ids,
                                 data_cache_path=data_cache_path)
        return dataset
    if name.find("train")!=-1:
        return build_dataset(0, 'train')
    if name.find("valid")!=-1:
        return build_dataset(1, 'valid')
    if name.find("test")!=-1:
        return build_dataset(2, 'test')

@dlp.log
def build_dataset(dataset_name, data_prefix, data_impl,
                  splits_string, num_samples,
                  seq_length, seed, skip_warmup,
                  *,
                  data_cache_path=None):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name, data_prefix[0], data_impl,
                                 splits_string, num_samples, seq_length,
                                 seed, skip_warmup,
                                 data_cache_path=data_cache_path)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output
        num_samples = sum(dataset_num_samples)

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i], data_impl,
                                splits_string, dataset_num_samples[i],
                                seq_length, seed, skip_warmup,
                                data_cache_path=data_cache_path)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights, num_samples,
                                       data_cache_path=data_cache_path)

    return dataset

@dlp.log
def _build_dataset(dataset_name, data_prefix, data_impl, splits_string,
                   num_samples, seq_length, seed, skip_warmup,
                   *,
                   data_cache_path=None):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    print_rank_0('    {}:'.format(dataset_name))
    print_rank_0('     document indices in [0, {}) total of {} '
                 'documents'.format(total_num_of_documents, total_num_of_documents))

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = GPTDataset(dataset_name, data_prefix, documents, indexed_dataset,
                         splits_string, num_samples, seq_length, seed,
                         data_cache_path=data_cache_path)

    return dataset

@dlp.log
def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPTDataset(torch.utils.data.Dataset):
    @dlp.log
    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 splits_string, num_samples, seq_length, seed,
                 return_doc_ids=False, *,
                 data_cache_path=None):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = \
            _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  splits_string, num_samples, seq_length, seed,
                                  data_cache_path=data_cache_path)


    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1
    @dlp.log
    def __getitem__(self, idx):
        args = get_args()
        orig_idx = idx
        # Get the shuffled index.
        try:
            idx = self.shuffle_idx[idx]
        except IndexError as exc:
            if is_rank_0():
                import json
                from rich import print_json
                print(exc)
                print(
                    '\n'.join(
                        ['-------------------------------------------------',
                         f'Trying to access {idx=} from self.shuffle_idx,',
                         f'but {len(self.shuffle_idx)=}',
                         '-------------------------------------------------']
                    )
                )
                print_json(
                    json.dumps(
                        {
                            'doc_idx': len(self.doc_idx),
                            'sample_idx': len(self.sample_idx),
                            'shuffle_idx': len(self.shuffle_idx),
                        },
                        indent=4,
                    )
                )
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        text_name = 'text'
        if args.use_dataset_only:
            text_name = 'input_ids'
        sample_dict = {text_name: np.array(sample, dtype=np.int64)}
        if args.return_data_index:
            sample_dict.update({'index': np.array([orig_idx], dtype=np.int64)})

        if self.return_doc_ids: # for retro preprocessing
            sample_dict.update({'doc_ids': np.array(doc_ids, dtype=np.int64)})

        if args.use_dataset_only:
            sample_dict.update({'labels': np.array(sample, dtype=np.int64)})

        return sample_dict

@dlp.log
def _build_index_mappings(name, data_prefix, documents, sizes,
                          splits_string, num_samples, seq_length, seed,
                          *,
                          data_cache_path):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    args = get_args()
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    if args.train_data_exact_num_epochs is not None and name == 'train':
        num_epochs = args.train_data_exact_num_epochs

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Number of epochs {num_epochs}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + '_doc_idx.npy'
    sample_idx_filename = desc_hash + '_sample_idx.npy'
    shuffle_idx_filename = desc_hash + '_shuffle_idx.npy'

    if name == 'train':
        # force to use certain index files
        if args.train_desc_path is not None:
            desc_filename = args.train_desc_path
        if args.train_doc_idx_path is not None:
            doc_idx_filename = args.train_doc_idx_path
        if args.train_sample_idx_path is not None:
            sample_idx_filename = args.train_sample_idx_path
        if args.train_shuffle_idx_path is not None:
            shuffle_idx_filename = args.train_shuffle_idx_path

    # Look for cache in main data dir first to avoid unnecessary
    # duplication, then look in data-cache-path if specified,
    # If nothing is found, use the last path looked in
    build_indices = True
    prefixes = [os.path.join(os.path.dirname(data_prefix), 'index-cache')]
    if data_cache_path is not None:
        prefixes.append(data_cache_path)
    for prefix in prefixes:
        idx_path = {
            'desc': os.path.join(prefix, desc_filename),
            'doc': os.path.join(prefix, doc_idx_filename),
            'sample': os.path.join(prefix, sample_idx_filename),
            'shuffle': os.path.join(prefix, shuffle_idx_filename)
        }
        for f in idx_path.values():
            if not os.path.isfile(f):
                break
        else:
            # Found our files!
            build_indices = False
            break
    data_cache_dir = os.path.dirname(idx_path['desc'])
    data_cache_success = True

    # Build the indexed mapping if not exist.
    if build_indices:
        # Since this function will be called by all the rank in the very beginning. Therefore, we assume that all the 
        # ranks will first create the document files, and then read it. 
        # There will not be contension effects going on either
        print_rank_0(f" > WARNING: could not find index map files, building on rank {torch.distributed.get_rank()}")

        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.

        # If we need only one epoch, then separating last epoch  does
        # not mean anything.
        if num_epochs == 1:
            separate_last_epoch = False
            print_rank_0(' > only one epoch required, setting '
                  'separate_last_epoch to False')

        else:
            # Get the number of samples for the last epoch
            num_samples_from_epochs_minus_one = (
                (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            last_epoch_num_samples = num_samples - \
                                     num_samples_from_epochs_minus_one
            assert last_epoch_num_samples >= 0, \
                'last epoch number of samples should be non-negative.'
            num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            assert last_epoch_num_samples <= (num_samples_per_epoch + 1), \
                'last epoch number of samples exceeded max value.'
            # If we have less than 80% of the samples for the last epoch,
            # seperate out the epoch and treat it differently.
            # Note: the 80% number is just based on common sense and can
            # be adjusted if needed.
            separate_last_epoch = (last_epoch_num_samples <
                                   int(0.80 * num_samples_per_epoch))
            if separate_last_epoch:
                string = ' > last epoch number of samples ({}) is smaller '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to True'
            else:
                string = ' > last epoch number of samples ({}) is larger '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to False'
            print_rank_0(string.format(last_epoch_num_samples,
                                num_samples_per_epoch))


        try:
            os.makedirs(data_cache_dir, exist_ok=True)

            # description
            with open(idx_path['desc'], 'wt') as fd:
                fd.write(desc)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(idx_path['doc'], doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch, torch.distributed.get_rank()==0)
            np.save(idx_path['sample'], sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(idx_path['shuffle'], shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))
        except OSError:
            print(f'There was an error trying to create the data cache directory ({data_cache_dir})')
            print('or a file in it. This defaults to a directory "index-cache" within the directory')
            print('the data files are in and can be set with the --data-cache-path argument. Please')
            print('ensure you have write access to this directory or specify one that you do have')
            print('write access to.')
            data_cache_success = False

    # Load mappings.
    start_time = time.time()
    print_rank_0(f" > loading doc-idx mapping from {idx_path['doc']}")
    doc_idx = np.load(idx_path['doc'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading sample-idx mapping from {idx_path['sample']}")
    sample_idx = np.load(idx_path['sample'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading shuffle-idx mapping from {idx_path['shuffle']}")
    shuffle_idx = np.load(idx_path['shuffle'], allow_pickle=True, mmap_mode='r')

    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, desc, desc_hash


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs

@dlp.log
def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))

@dlp.log
def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx

@dlp.log
def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print_rank_0(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size))

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

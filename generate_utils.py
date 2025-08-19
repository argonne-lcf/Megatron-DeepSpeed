

"""Generate function post training"""

import os
from rich import print
import torch
import math
import numpy as np
import time
from datetime import datetime
import threading

from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
#from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import (
    report_memory,
    throughput_calculator,
    checkpoint_throughput_calculator
)
from pathlib import Path

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import subprocess
import wandb

from torch import nn
import torch.nn.functional as F

# from ezpz import get_logger
from ezpz.dist import get_world_size, setup_wandb, get_rank

# More imports
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.training import print_datetime, _create_ds_config_dict
from megatron.training import setup_model_and_optimizer
from megatron.training import load_model_weights_only, get_model
from megatron.training import get_optimizer_param_scheduler, cyclic_iter
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import load_checkpoint
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.arguments import core_transformer_config_from_args
from megatron import update_num_microbatches
from megatron import get_num_microbatches
from megatron.utils import throughput_calculator, get_parameters_in_billions
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process
from megatron.text_generation.forward_step import ForwardStep, InferenceParams
from megatron.text_generation.sampling import sample
from megatron.text_generation.tokenization import detokenize_generations
from megatron.text_generation.communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage)
from megatron.checkpointing import save_checkpoint
from megatron.utils import get_ltor_masks_and_position_ids


def generate_post_training(
        model, prompts, tokens_to_generate,
        top_k = 0,
        top_p = 1.0,
        temperature = 1.0,
        top_p_decay=0.0,
        top_p_bound=0.0,
        add_BOS=False,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        prevent_newline_after_colon=False,
        random_seed=42,
        return_output_log_probs = False,
        fprint=True
        ):

    print_rank_0(f'Generation mode..')
    model[0].eval()
    
    args = get_args()
    print_rank_0(f'Seq length in args: {args.seq_length}')

    tokenizer = get_tokenizer()
    print_rank_0(f'Number of elements in tokenizer vocab: {len(tokenizer.vocab)}')
    # prompts=["A sequence", "A sequence","A sequence", "A sequence", "A sequence"]
    # tokens_to_generate = 64

    # add_BOS = False
    if add_BOS:
        prompts_tokens = [[tokenizer.eod] + tokenizer.tokenize(prompt)
                        for prompt in prompts]
    else:
        prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

    if fprint: print_rank_0(f'prompts_tokens: {prompts_tokens}')

    # Make all tokenized prompts to be of same length as max length of the prompts
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    max_prompt_len = max(prompts_length)
    samples_length = max_prompt_len + tokens_to_generate
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([tokenizer.eod] * padding_size)
    print(f"prompts_length:{prompts_length}, max_prompt_len:{max_prompt_len}, samples_length:{samples_length}")
    # Now we are in a structured format, we can convert to tensors
    prompts_tokens_tensor = get_accelerator().LongTensor(prompts_tokens) #torch.cuda.LongTensor(prompts_tokens)
    prompts_length_tensor = get_accelerator().LongTensor(prompts_length) #torch.cuda.LongTensor(prompts_length)
    if fprint:
        print_rank_0(f'prompts_tokens_tensor: {prompts_tokens_tensor}')
        print_rank_0(f'prompts_length_tensor: {prompts_length_tensor}')

    # Getting attributes to set inference_params
    batch_size = prompts_tokens_tensor.size(0)
    min_prompt_length = prompts_length_tensor.min().item()
    max_sequence_length = prompts_tokens_tensor.size(1)

    if fprint:
        print_rank_0(f'batch_size: {batch_size}')
        print_rank_0(f'min_prompt_length: {min_prompt_length}')
        print_rank_0(f'max_sequence_length: {max_sequence_length}')
        print_rank_0(f'max_position_embeddings: {args.max_position_embeddings}')
        print_rank_0(f'args.max_tokens_to_oom: {args.max_tokens_to_oom}')

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

    # INSTANTIATING FORWARD_STEP ?
    # model_fwd = ForwardStep(model[0], batch_size, max_sequence_length)
    inference_params = InferenceParams(batch_size,
                                        max_sequence_length)

    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
        print_rank_0(f'args.eos_id: {args.eos_id}')
    else:
        termination_id = tokenizer.eod
        print_rank_0(f'tokenizer.eod: {tokenizer.eod}')

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None

    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                        dtype=torch.float32,
                                        device=torch.cuda.current_device())
            if fprint: print_rank_0(f'On mpu.is_pipeline_last_stage branch and output_log_probs is set: {output_log_probs}')
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                #device=torch.cuda.current_device()) * max_sequence_length
                device=get_accelerator().current_device_name()) * max_sequence_length
        if fprint: print_rank_0(f'On mpu.is_pipeline_last_stage branch and generated_sequence_lengths: {generated_sequence_lengths}')

    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                    device=get_accelerator().current_device_name())

    print(f"here : prompts_tokens_tensor:{prompts_tokens_tensor}, shape:{prompts_tokens_tensor.shape}")
    with torch.no_grad():
        prompts_attention_mask, _, prompts_position_ids = get_ltor_masks_and_position_ids(
                                                    data=prompts_tokens_tensor,
                                                    eod_token=None,
                                                    reset_position_ids=False,
                                                    reset_attention_mask=False,
                                                    eod_mask_loss=False
                                                    )
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):
            # Pick the slice that we need to pass through the network.
            print(f"prev_context_length:{prev_context_length}, context_length:{context_length}, prompts_tokens_tensor:{prompts_tokens_tensor}")
            #tokens2use = prompts_tokens_tensor[:, prev_context_length:context_length]
            tokens2use = prompts_tokens_tensor[:, :context_length]
            #positions2use = prompts_position_ids[:, prev_context_length:context_length]
            positions2use = prompts_position_ids[:, :context_length]
            attention_mask2use = prompts_attention_mask[
                ..., :context_length, :context_length]
                #..., prev_context_length:context_length, :context_length]

            # #logits will be meanigful only in the last pipeline stage.
            if fprint:
                print_rank_0(f'tokens2use shape: {tokens2use.size()}')
                print_rank_0(f'positions2use shape: {positions2use.size()}')
                print_rank_0(f'attention_mask2use shape: {attention_mask2use.size()}')
                print_rank_0(f'prompts_tokens_tensor shape: {prompts_tokens_tensor.size()}')
                print_rank_0(f'prompts_position_ids shape: {prompts_position_ids.size()}')
                print_rank_0(f'prompts_attention_mask shape: {prompts_attention_mask.size()}')

            # ------
            # plogits = forward_step(tokens2use, positions2use, attention_mask2use)
            # plogits = plogits[0]
            # print_rank_0(f'context_length: {context_length}, plogits: {plogits}')

            # plogits = model[0](prompts_tokens_tensor, 
            #                     prompts_position_ids, 
            #                     prompts_attention_mask, 
            #                     inference_params=inference_params
            #                 )
            # print_rank_0(f'logits: {plogits}')
            #-------

            # Changing seq length in inference params dynamically
            inference_params = InferenceParams(batch_size,
                                        tokens2use.size(1))
            print(f"tokens2use: {tokens2use}, positions2use:{positions2use}, attention_mask2use:{attention_mask2use}")
            plogits = model[0](tokens2use, 
                                positions2use, 
                                attention_mask2use, 
                                inference_params=inference_params
                            )
            plogits = plogits[0]
            # plogits = torch.cuda.FloatTensor(plogits)
            if fprint:
                print_rank_0(f'plogits: {plogits.size()}')
                print_rank_0(f'plogits type: {plogits.dtype}')

            if mpu.is_pipeline_last_stage():
                if prevent_newline_after_colon:
                    plogits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
                # Always the last stage should have an output.
                assert plogits is not None

                # Sample.
                last_token_logits = plogits[:, -1, :]
                new_sample = sample(last_token_logits,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    vocab_size=tokenizer.vocab_size)
                if top_p > 0.0 and top_p_decay > 0.0:
                    top_p = top_p * top_p_decay
                    if top_p_bound > 0.0:
                        top_p = max(top_p, top_p_bound)

                if fprint:
                    print_rank_0(f'new_sample: {new_sample}')
                    for nidx, ns in enumerate(new_sample.cpu().numpy().tolist()):
                        print_rank_0(f'nidx: {nidx}, new_sample[{nidx}]: {tokenizer.detokenize(ns)}')
                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = prompts_length_tensor <= context_length
                # Update the tokens.
                if fprint: 
                    print_rank_0(f'started: {started}')
                    # print_rank_0(f'prompts_tokens_tensor before copying new_sample: {prompts_tokens_tensor}')
                    for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                        print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor before[{nidx}]: {tokenizer.detokenize(ns)}')

                prompts_tokens_tensor[started, context_length] = new_sample[started]
                if fprint:
                    # print_rank_0(f'prompts_tokens_tensor after copying new_sample: {prompts_tokens_tensor}')
                    for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                        print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor after[{nidx}]: {tokenizer.detokenize(ns)}')

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                prompts_tokens_tensor[:, context_length])
            # for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
            #     print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor after copy_from_last_to_first_pipeline_stage [{nidx}]: {tokenizer.detokenize(ns)}')

            # Update the context length for the next token generation.
            prev_context_length = context_length
            if fprint: print_rank_0(f'prev_context_length: {prev_context_length}')

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                # These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                if stop_on_double_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_two_eols
                elif stop_on_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_eol = (new_sample == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_eol
                else:
                    done_token = (new_sample == termination_id).byte() & \
                        started.byte()

                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                    tensor=done)
            if use_eod_token_for_early_termination and done:
                print_rank_0(f'done: {done}')
                break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================
    # for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
    #     print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop [{nidx}]: {tokenizer.detokenize(ns)}')
    prompts_tokens_tensor = prompts_tokens_tensor[:, :(context_length + 1)]
    # for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
    #     print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and slicing with ctx length[{nidx}]: {tokenizer.detokenize(ns)}')
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    # if fprint:
    #     for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
    #         print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and befoer final post-process[{nidx}]: {tokenizer.detokenize(ns)}')
    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        prompts_plus_generations = []

        if fprint:
            for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and after final post-process[{nidx}]: {tokenizer.detokenize(ns)}')

        rtokens = prompts_tokens_tensor.cpu().numpy().tolist()
        rlengths = prompts_length_tensor.cpu().numpy().tolist()
        if fprint: print_rank_0(f'rlengths: {rlengths}')
        # for sequence_tokens, slength in zip(rtokens, rlengths):
        for sequence_tokens in rtokens:
            # sequence_tokens = sequence_tokens[:slength]
            prompts_plus_generations.append(
                tokenizer.detokenize(sequence_tokens))
        # _, prompts_plus_generations, prompts_plus_generations_segments = \
        #     detokenize_generations(prompts_tokens_tensor, prompts_length_tensor, True)

    for prompt, prompt_response in zip(prompts, prompts_plus_generations):
        print_rank_0(f'------------------')
        print_rank_0(f'prompt: {prompt}')
        print_rank_0(f'prompt and response: {prompt_response}')

    return prompts_plus_generations

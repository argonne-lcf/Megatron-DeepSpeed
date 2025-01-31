from argparse import Namespace
import os
from pathlib import Path
from typing import Optional

import ezpz
import torch
import torch.distributed
import deepspeed

from transformers import AutoModelForCausalLM

logger = ezpz.get_logger(__name__)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct'
    )
    parser.add_argument('--device', type=str, default=None, required=False)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--zero-stage', type=int, default=3)
    # add arg for output directory
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--kv-offload', action='store_true')
    parser.add_argument('--async-kv-offload', action='store_true')
    parser.add_argument('--gen-len', type=int, default=1024)
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args()


def meta_to_cpu(container, dtype=None):
    if isinstance(container, torch.Tensor):
        return torch.empty(*container.shape, dtype=dtype or container.dtype)
    elif isinstance(container, tuple):
        return tuple(meta_to_cpu(x, dtype) for x in container)
    elif isinstance(container, dict):
        return dict((k, meta_to_cpu(v, dtype)) for k, v in container.items())
    else:
        raise ValueError(f'Invalid type: {container}')


def get_model(
    model_name: str = 'meta-llama/Llama-3.2-1B-Instruct',
    dummy: Optional[bool] = None,
    ignore_mismatched_sizes: bool = True,
) -> torch.nn.Module:
    if dummy:
        filename = Path('.').joinpath(
            f'{model_name}.replace("/", "-")-hf-weights'
        )
        if not filename.exists():
            from accelerate import init_empty_weights

            logger.info('Creating dummy weights')
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    f'{model_name}',
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )
            model.save_pretrained(
                filename,
                state_dict=meta_to_cpu(model.state_dict(), torch.float16),
            )
            return model

    model = AutoModelForCausalLM.from_pretrained(
        f'{model_name}',
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    return model


def get_ds_config(
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    zero_stage: int = 3,
    hidden_size: Optional[int] = None,
) -> dict:
    train_batch_size = (
        micro_batch_size * ezpz.get_world_size() * gradient_accumulation_steps
    )
    zero_config = {
        'stage': zero_stage,
    }
    if zero_stage == 3:
        if hidden_size is not None:
            zero_config |= {
                'stage3_prefetch_bucket_size': 2 * hidden_size * hidden_size,
                'stage3_param_persistence_threshold': hidden_size,
                'stage3_max_live_parameters': 2 * hidden_size * hidden_size,
            }
        zero_config |= {
            'offload_optimizer': {
                'device': 'cpu',
            },
            'offload_param': {
                'device': 'cpu',
            },
        }

    return {
        'bf16': {'enabled': True},
        'fp16': {'enabled': False},
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'optimizer': {
            'type': 'Adam',
        },
        'steps_per_print': 1,
        'train_batch_size': train_batch_size,
        'train_micro_batch_size_per_gpu': 1,
        'wall_clock_breakdown': True,
        'zero_optimization': zero_config,
    }


def convert_checkpoint(args: Namespace):
    if args.device is not None and args.device == 'cpu':
        os.environ['TORCH_DEVICE'] = 'cpu'
        os.environ['DS_ACCELERATOR'] = 'cpu'

    if args.zero_stage == 3:
        cm = deepspeed.zero.Init()
    else:
        from contextlib import nullcontext

        cm = nullcontext()

    with cm:
        with torch.no_grad():
            model = get_model(
                args.model, ignore_mismatched_sizes=not args.strict
            )

    assert isinstance(model, torch.nn.Module)
    if args.kv_offload:
        model.set_kv_cache_offload(
            True,
            gen_len=args.gen_len,
            async_kv_offload=args.async_kv_offload,
        )

    logger.info(f'model:\n{model}')
    logger.info(f'{model.config=}')
    ds_config = get_ds_config(
        args.train_batch_size,
        args.zero_stage,
        hidden_size=model.config.hidden_size,
    )
    output_dir = Path('zero-checkpoints').joinpath(
        f'{args.model}-zs{args.zero_stage}-mb{args.train_batch_size}'
    )

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    logger.info(f'Saving ZeRO checkpoint to {output_dir}')

    ds_engine.save_checkpoint(output_dir)

    torch.distributed.barrier()


def main():
    _ = ezpz.setup_torch(backend='DDP')
    args = parse_args()
    convert_checkpoint(args)


if __name__ == '__main__':
    main()

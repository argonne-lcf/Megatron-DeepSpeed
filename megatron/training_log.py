# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
"""
training_log.py
"""

import logging
import os

from deepspeed import get_accelerator
import ezpz as ez
import torch

from megatron.core import mpu
from megatron.global_vars import (
    get_args,
    get_num_microbatches,
    get_tensorboard_writer,
    get_timers,
)
from megatron.utils import (
    Profile,
    is_last_rank,
    report_memory,
    throughput_calculator,
    num_floating_point_operations,
)

import csv

RANK: int = ez.get_rank()
WORLD_SIZE: int = ez.get_world_size()
DEVICE_TYPE: str = ez.dist.get_torch_device_type()
DEVICE: torch.device = torch.device(DEVICE_TYPE)

log: logging.Logger = logging.getLogger(__name__)
LOG_LEVEL: str = str(os.environ.get("LOG_LEVEL", "INFO")).upper()
log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


dlp = Profile("TRAINING_LOG")


@dlp.log
def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    model=None,
    optimizer=None,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    accelerator = get_accelerator()
    timers = get_timers()
    writer = get_tensorboard_writer()
    assert args is not None and timers is not None and accelerator is not None
    wandb_metrics = {}
    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, accelerator.FloatTensor([0.0]))
                + loss_dict[key]
            )
        else:
            try:
                value = loss_dict[key].float().sum().item()
            except AttributeError:
                value = loss_dict[key]
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )
    
    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "grads-all-reduce",
        "grads-reduce-scatter",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )
    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and (
        iteration % args.tensorboard_log_interval == 0
    ):
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        writer.add_scalar(
            "steps-vs-samples/y=steps,x=samples", iteration, args.consumed_train_samples
        )
        writer.add_scalar(
            "steps-vs-samples/y=samples,x=steps", args.consumed_train_samples, iteration
        )
        writer.add_scalar(
            "steps-vs-tokens/y=steps,x=tokens", iteration, args.consumed_train_tokens
        )
        writer.add_scalar(
            "steps-vs-tokens/y=tokens,x=steps", args.consumed_train_tokens, iteration
        )
        if args.log_learning_rate_to_tensorboard:
            wandb_metrics |= {
                "learning-rate/iteration": iteration,
                "learning-rate/learning-rate": learning_rate,
            }
            writer.add_scalar("learning-rate/learning-rate", learning_rate, iteration)
            writer.add_scalar(
                "learning-rate/learning-rate vs samples",
                learning_rate,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "learning-rate/learning-rate vs tokens",
                learning_rate,
                args.consumed_train_tokens,
            )
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size/batch-size", batch_size, iteration)
            writer.add_scalar(
                "batch-size/batch-size vs samples",
                batch_size,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "batch-size/batch-size vs tokens",
                batch_size,
                args.consumed_train_tokens,
            )
        wandb_metrics |= {
            "lm-loss-training/iteration": iteration,
            "lm-loss-training/consumed_train_tokens": args.consumed_train_tokens,
        }
        for key in loss_dict:
            wandb_metrics |= {f"lm-loss-training/{key}": loss_dict[key]}
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(
                f"lm-loss-training/{key}" + " vs samples",
                loss_dict[key],
                args.consumed_train_samples,
            )
            writer.add_scalar(
                f"lm-loss-training/{key}" + " vs tokens",
                loss_dict[key],
                args.consumed_train_tokens,
            )
        if args.fp16 and loss_scale and args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale/loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale/loss-scale vs samples",
                loss_scale,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "loss-scale/loss-scale vs tokens",
                loss_scale,
                args.consumed_train_tokens,
            )
        if args.log_world_size_to_tensorboard:
            writer.add_scalar("world-size/world-size", args.world_size, iteration)
            writer.add_scalar(
                "world-size/world-size vs samples",
                args.world_size,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "world-size/world-size vs tokens",
                args.world_size,
                args.consumed_train_tokens,
            )
        if grad_norm is not None:
            wandb_metrics |= {"training/grad-norm": grad_norm}
            writer.add_scalar("grad-norm/grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm/grad-norm vs samples", grad_norm, args.consumed_train_samples
            )
            writer.add_scalar(
                "grad-norm/grad-norm vs tokens", grad_norm, args.consumed_train_tokens
            )
        if num_zeros_in_grad is not None:
            wandb_metrics |= {"training/num-zeros": num_zeros_in_grad}
            writer.add_scalar("num-zeros/num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros/num-zeros vs samples",
                num_zeros_in_grad,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "num-zeros/num-zeros vs tokens",
                num_zeros_in_grad,
                args.consumed_train_tokens,
            )
        if params_norm is not None:
            wandb_metrics |= {"training/params-norm": params_norm}
            writer.add_scalar("params-norm/params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm/params-norm vs samples",
                params_norm,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "params-norm/params-norm vs tokens",
                params_norm,
                args.consumed_train_tokens,
            )
            ### AG Debug ###
            print("------- param norm --------")
            print(params_norm)
            print("------- param norm --------")
            ### AG Debug ###
        if hasattr(args, "actual_seq_length"):
            writer.add_scalar(
                "seqlen/actual_seq_length", args.actual_seq_length, iteration
            )
            writer.add_scalar(
                "seqlen/actual_seq_length vs samples",
                args.actual_seq_length,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "seqlen/actual_seq_length vs tokens",
                args.actual_seq_length,
                args.consumed_train_tokens,
            )
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar(
                "seqlen/curriculum_seqlen", args.curriculum_seqlen, iteration
            )
            writer.add_scalar(
                "seqlen/curriculum_seqlen vs samples",
                args.curriculum_seqlen,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "seqlen/curriculum_seqlen vs tokens",
                args.curriculum_seqlen,
                args.consumed_train_tokens,
            )
        if args.random_ltd:
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length",
                args.random_ltd_reserved_length,
                iteration,
            )
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length vs samples",
                args.random_ltd_reserved_length,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length vs tokens",
                args.random_ltd_reserved_length,
                args.consumed_train_tokens,
            )
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group["params"]):
                    state_param = getattr(optimizer, "state", None)
                    if state_param is not None:
                        exp_avg_sq = state_param.get("exp_avg_sq", torch.tensor(0.0))
                        exp_avg = state_param.get("exp_avg", torch.tensor(0.0))
                        opt_stats[0] += (torch.norm(exp_avg_sq).item()) ** 2
                        opt_stats[1] += (torch.norm(exp_avg_sq.sqrt()).item()) ** 2
                        opt_stats[2] += (torch.norm(exp_avg).item()) ** 2
                        opt_stats[3] += (torch.norm(param).item()) ** 2
                        opt_stats[4] += torch.norm(exp_avg_sq, p=1).item()
                        opt_stats[5] += torch.norm(exp_avg_sq.sqrt(), p=1).item()
                        opt_stats[6] += torch.norm(exp_avg, p=1).item()
                        opt_stats[7] += torch.norm(param, p=1).item()
                        opt_stats_2[0] = max(
                            opt_stats_2[0],
                            abs(exp_avg_sq.max().item()),
                            abs(exp_avg_sq.min().item()),
                        )
                        opt_stats_2[1] = max(
                            opt_stats_2[1], exp_avg_sq.sqrt().abs_().max().item()
                        )
                        opt_stats_2[2] = max(
                            opt_stats_2[2],
                            abs(exp_avg.max().item()),
                            abs(exp_avg.min().item()),
                        )
                        opt_stats_2[3] = max(
                            opt_stats_2[3],
                            abs(param.max().item()),
                            abs(param.min().item()),
                        )
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                # opt_stats = opt_stats.clone().detach()
                # opt_stats = get_accelerator().FloatTensor
                opt_stats = accelerator.FloatTensor(opt_stats)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_sequence_data_parallel_group()
                )
                # opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                # opt_stats_2 = opt_stats_2.clone().detach()
                opt_stats_2 = accelerator.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_sequence_data_parallel_group(),
                )

            if args.tensor_model_parallel_size > 1:
                # opt_stats = opt_stats.clone().detach()
                opt_stats = accelerator.FloatTensor(opt_stats)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_tensor_model_parallel_group()
                )
                # opt_stats_2 = opt_stats_2.clone().detach()
                opt_stats_2 = accelerator.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group(),
                )

            if args.pipeline_model_parallel_size > 1:
                # opt_stats = opt_stats.clone().detach()
                opt_stats = accelerator.FloatTensor(opt_stats)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_pipeline_model_parallel_group()
                )
                # opt_stats_2 = opt_stats_2.clone().detach()
                opt_stats_2 = accelerator.FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group(),
                )
            wandb_metrics |= {
                "optimizer/learning_rate": learning_rate,
                "optimizer/iteration": args.iteration,
                "optimizer/consumed_train_tokens": args.consumed_train_tokens,
                "optimizer/variance_l2": opt_stats[0] ** 0.5,
                "optimizer/variance_sqrt_l2": opt_stats[1] ** 0.5,
                "optimizer/momentum_l2": opt_stats[2] ** 0.5,
                "optimizer/weight_l2": opt_stats[3] ** 0.5,
                "optimizer/variance_l1": opt_stats[4],
                "optimizer/variance_sqrt_l1": opt_stats[5],
                "optimizer/momentum_l1": opt_stats[6],
                "optimizer/weight_l1": opt_stats[7],
                "optimizer/variance_abs_max": opt_stats_2[0],
                "optimizer/variance_sqrt_abs_max": opt_stats_2[1],
                "optimizer/momentum_abs_max": opt_stats_2[2],
                "optimizer/weight_abs_max": opt_stats_2[3],
            }
            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar(
                    "optimizer/variance_l2 vs tokens",
                    opt_stats[0] ** 0.5,
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l2 vs tokens",
                    opt_stats[1] ** 0.5,
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_l2 vs tokens",
                    opt_stats[2] ** 0.5,
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_l2 vs tokens",
                    opt_stats[3] ** 0.5,
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_l1 vs tokens",
                    opt_stats[4],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l1 vs tokens",
                    opt_stats[5],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_l1 vs tokens",
                    opt_stats[6],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_l1 vs tokens",
                    opt_stats[7],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_abs_max vs tokens",
                    opt_stats_2[0],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_abs_max vs tokens",
                    opt_stats_2[1],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_abs_max vs tokens",
                    opt_stats_2[2],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_abs_max vs tokens",
                    opt_stats_2[3],
                    args.consumed_train_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_l2", opt_stats[0] ** 0.5, iteration
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l2", opt_stats[1] ** 0.5, iteration
                )
                writer.add_scalar(
                    "optimizer/momentum_l2", opt_stats[2] ** 0.5, iteration
                )
                writer.add_scalar("optimizer/weight_l2", opt_stats[3] ** 0.5, iteration)
                writer.add_scalar("optimizer/variance_l1", opt_stats[4], iteration)
                writer.add_scalar("optimizer/variance_sqrt_l1", opt_stats[5], iteration)
                writer.add_scalar("optimizer/momentum_l1", opt_stats[6], iteration)
                writer.add_scalar("optimizer/weight_l1", opt_stats[7], iteration)
                writer.add_scalar(
                    "optimizer/variance_abs_max", opt_stats_2[0], iteration
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_abs_max", opt_stats_2[1], iteration
                )
                writer.add_scalar(
                    "optimizer/momentum_abs_max", opt_stats_2[2], iteration
                )
                writer.add_scalar("optimizer/weight_abs_max", opt_stats_2[3], iteration)

    assert args is not None
    assert timers is not None
    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, "actual_seq_length"):
            seq_len = args.actual_seq_length
        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
            model, args, elapsed_time, total_iterations
        )
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        tokens_per_gpu_per_second = tokens_per_sec / args.world_size
        tokens_per_gpu_per_second_per_replica = (
            tokens_per_gpu_per_second / args.data_parallel_size
        )
        # NOTE: [2024-06-19]
        # Updated to use (more accurate) calculation according to
        # `num_floating_point_operations` from NVIDIA/Megatron-LM
        num_flop_lm = num_floating_point_operations(args, batch_size)
        num_flop_per_sec_lm = num_flop_lm / elapsed_time_per_iteration
        tflops_lm = num_flop_per_sec_lm / (10**12)
        tflops_lm_per_gpu = tflops_lm / args.world_size
        wandb_metrics |= {
            "throughput/iteration-time": elapsed_time_per_iteration,  # 1000 ms / s
            "throughput/samples_per_sec": samples_per_sec,
            "throughput/samples_per_sec_per_replica": samples_per_sec_per_replica,
            "throughput/tokens_per_sec": tokens_per_sec,
            "throughput/tokens_per_sec_per_replica": tokens_per_sec_per_replica,
            "throughput/tokens_per_gpu_per_sec": tokens_per_gpu_per_second,
            "throughput/tokens_per_gpu_per_sec_per_replica": tokens_per_gpu_per_second_per_replica,
            "throughput/tflops": tflops,
            "throughput/tflops-new": num_flop_lm / elapsed_time_per_iteration,
            "throughput/tflops-lm": tflops_lm_per_gpu,
            "throughput/approx_params_in_billions": approx_parameters_in_billions,
            "throughput/elapsed_ms_per_iteration": elapsed_time_per_iteration,
            "throughput/iteration": iteration,
        }
        if loss_dict is not None:
            wandb_metrics |= {
                "loss/iteration": iteration,
                **{f"loss/{k}": v for k, v in loss_dict.items()},
            }
        if writer and args.log_timers_to_tensorboard:
            writer.add_scalar(
                "iteration-time/iteration-time", elapsed_time_per_iteration, iteration
            )
            writer.add_scalar(
                "iteration-time/iteration-time vs samples",
                elapsed_time_per_iteration,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "iteration-time/iteration-time vs tokens",
                elapsed_time_per_iteration,
                args.consumed_train_tokens,
            )
        # metrics_to_log = {
        #     'iteration': iteration,
        #     'train_iters': args.train_iters,
        #     'consumed_samples': args.consumed_train_samples,
        #     'consumed_tokens': args.consumed_tokens,
        # }
        log_string = f" iteration={iteration:8d}/{args.train_iters:8d} |"
        # .format( iteration, args.train_iters)
        log_string += (
            f" consumed_samples={args.consumed_train_samples:12d} |"
            # .format(args.consumed_train_samples)
        )
        log_string += f" consumed_tokens={args.consumed_train_tokens:12d} |"
        # .format( args.consumed_train_tokens)
        log_string += (
            " elapsed_time_per_iteration_ms="
            f"{elapsed_time_per_iteration * 1000.0:.1f} |"
            # .format( elapsed_time_per_iteration * 1000.0)
        )
        log_string += f" learning_rate={learning_rate:.6g} |"
        log_string += f" global_batch_size={batch_size:5d} |"
        # if wandb is not None and getattr(wandb, 'run', None) is not None:
        wandb_metrics |= {
            "training/iteration": iteration,
            "training/iteration_time": elapsed_time_per_iteration,
            "training/iteration_time_vs_tokens": (
                elapsed_time_per_iteration / args.consumed_train_tokens
            ),
            "training/iteration_time_vs_samples": (
                (elapsed_time_per_iteration / args.consumed_train_samples),
            ),
            "training/consumed_samples": args.consumed_train_samples,
            "training/consumed_tokens": args.consumed_train_tokens,
        }
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}={:.6f} |".format(key, avg)
                total_loss_dict[key] = accelerator.FloatTensor([0.0])
        if loss_scale is not None:
            log_string += " loss_scale={:.1f} |".format(loss_scale)
            wandb_metrics |= {"loss/loss_scale": loss_scale}
        if grad_norm is not None:
            log_string += " grad_norm={:.3f} |".format(grad_norm)
            wandb_metrics |= {"loss/grad_norm": grad_norm}
        if num_zeros_in_grad is not None:
            log_string += " num_zeros={:.1f} |".format(num_zeros_in_grad)
            wandb_metrics |= {"loss/num_zeros_in_grad": num_zeros_in_grad}
        if params_norm is not None:
            log_string += " params_norm={:.3f} |".format(params_norm)
            wandb_metrics |= {"loss/params_norm": params_norm}
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            log_string += " curriculum_seqlen={:5d} |".format(args.curriculum_seqlen)
        if args.random_ltd:
            log_string += " random_ltd reserved_length={:5d} |".format(
                args.random_ltd_reserved_length
            )
            # log_string += " | ".join([
            #     f"{seq_len=:5d} ",
            #     f"{}"
            #     f"number_of_skipped_iterations={:3d}",
            #
            # ])
        log_string += " actual_seqlen={:5d} |".format(seq_len)
        log_string += " number_of_skipped_iterations={:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number_of_nan_iterations={:3d} |".format(
            total_loss_dict[nan_iters_key]
        )
        log_string += " samples_per_second={:.3f} |".format(samples_per_sec)
        log_string += " tokens_per_gpu_per_second_tgs={:.3f} |".format(
            tokens_per_gpu_per_second
        )
        log_string += " [LM]TFLOPs={:.2f} |".format(tflops_lm_per_gpu)
        log_string += " [DS]TFLOPs={:.2f} |".format(tflops)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb_metrics |= {
                "training/skiped_iterations": total_loss_dict[skipped_iters_key]
            }
            wandb_metrics |= {"training/nan_iterations": total_loss_dict[nan_iters_key]}
            wandb.log(wandb_metrics)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        # print_rank_last(log_string)
        log.info(log_string)
        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        if timers is not None:
            timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag

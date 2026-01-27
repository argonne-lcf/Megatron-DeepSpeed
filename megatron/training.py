# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
"""Pretrain utilities."""
import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

from collections import OrderedDict
from datetime import datetime
import json
import logging
import math
import os
import sys
import time

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import (
    convert_to_random_ltd,
)
import ezpz as ez
import torch
import torch.distributed as tdist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import wandb
from megatron import (
    get_args,
    get_current_global_batch_size,
    get_num_microbatches,
    get_signal_handler,
    get_tensorboard_writer,
    get_timers,
    is_last_rank,
    update_num_microbatches,
)
from megatron.arguments import core_transformer_config_from_args
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
    write_args_to_tensorboard,
)
from megatron.model import Float16Module, GPTModel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.optimizer import get_megatron_optimizer
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training_log import training_log
from megatron.utils import (
    PerfTrace,
    Profile,
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    checkpoint_throughput_calculator,
    found_kill_switch,
    unwrap_model,
    update_rotary_pos_emb,
    mup_coord_check, ### Begin MuP Change ###
)

from megatron.profiler import (
    setup_profiler,
    trigger,
    on_step_begin,
    on_step_end,
)


dlp = Profile("TRAINING")

# from deepspeed import comm as dist

RANK: int = ez.get_rank()
WORLD_SIZE: int = ez.get_world_size()
# DEVICE_TYPE: str = ez.get_torch_device()
DEVICE_TYPE: str = ez.dist.get_torch_device_type()
DEVICE: torch.device = torch.device(DEVICE_TYPE)

log: logging.Logger = logging.getLogger(__name__)
LOG_LEVEL: str = str(os.environ.get("LOG_LEVEL", "INFO")).upper()
log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    tdist.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("[" + string + "] datetime={} ".format(time_str))


"""
Since v0.9.0, deepspeed.initialize() has forbidden simultaneous setting of args.deepspeed_config (Path) and ds_config dict.
So, we use ds_config dict which is the more flexible option
"""


def _create_ds_config_dict():
    args = get_args()
    assert args is not None
    if isinstance(args.deepspeed_config, dict):
        ds_config_dict = args.deepspeed_config
    else:
        with open(args.deepspeed_config, "r", encoding="utf-8") as config_file:
            ds_config_dict = json.load(config_file)
    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}
    # Clear config path
    args.deepspeed_config = None
    return ds_config_dict


@dlp.log
def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    data_post_process=None,
    external_args={},
) -> list[torch.nn.Module]:
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.

    Returns:
        model (torch.nn.Module)
    """
    
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        external_args=external_args,
    )
    args = get_args()
    assert args is not None
    if found_kill_switch():
        print_datetime(f"Detected kill switch at {args.kill_switch_file}. Exiting")
        sys.exit()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    # if get_accelerator().device_name() == "cuda":
    if DEVICE_TYPE == "cuda" and torch.cuda.is_available():
        set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    before_allreduce = time.time()
    global _TRAIN_START_TIME
    log.info(
        f"time to finish initialize_megatron: {time.time() - _TRAIN_START_TIME} seconds"
    )
    # start_time_tensor = DEVICE.DoubleTensor([_TRAIN_START_TIME])
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.double, device=DEVICE_TYPE
    )
    tdist.all_reduce(start_time_tensor, op=tdist.ReduceOp.MIN)
    log.info(f"allreduce call time: {time.time()-before_allreduce} seconds")
    _TRAIN_START_TIME = start_time_tensor.item()
    log.info(
        "time to initialize megatron (seconds)={:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")
    if os.getenv("DLIO_PROFILER_DATASET_DIR") is not None:
        extra_trace_path = os.environ["DLIO_PROFILER_DATASET_DIR"]
    else:
        extra_trace_path = ""
    os.makedirs(args.trace_dir, exist_ok=True)
    PerfTrace.initialize_log(
        f"{args.trace_dir}/trace-{ez.get_rank()}-of-{ez.get_world_size()}.pfw",
        f"{args.data_cache_path}:{extra_trace_path}:{args.data_path}:{args.save}:{args.load}",
        process_id=ez.get_rank(),
    )
    timers = get_timers()
    assert args is not None
    assert timers is not None
    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        if (
            "curriculum_learning" in args.deepspeed_config_dict
            and "enabled" in args.deepspeed_config_dict["curriculum_learning"]
        ):
            args.curriculum_learning_legacy = args.deepspeed_config_dict[
                "curriculum_learning"
            ]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler import (
                CurriculumScheduler,
            )

            args.curriculum_scheduler = CurriculumScheduler(
                args.deepspeed_config_dict["curriculum_learning"]
            )
        if "compression_training" in args.deepspeed_config_dict:
            args.compression_training = True

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider,
        model_type,
        teacher=False,
        data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider,
    )
    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate " "scheduler are built")
    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [
            data_iterators[0] for data_iterators in all_data_iterators
        ]
        valid_data_iterator = [
            data_iterators[1] for data_iterators in all_data_iterators
        ]
        test_data_iterator = [
            data_iterators[2] for data_iterators in all_data_iterators
        ]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
    if args.data_efficiency_curriculum_learning:
        if args.deepspeed_dataloader is not None:
            # We use args to pass the deepspeed_dataloader because adding
            # output to setup_model_and_optimizer will break the API for other
            # cases. We clear args.deepspeed_dataloader after updating
            # train_data_iterator because args will be saved in checkpoint and
            # attempting to save the whole deepspeed_dataloader will lead to
            # "AttributeError: Can't pickle local object...".
            train_data_iterator = iter(args.deepspeed_dataloader)
            args.deepspeed_dataloader = None
        else:
            train_data_iterator = None
    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")
    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None
    if args.mos or args.kd:  # Set up teacher model
        args.teacher_model = setup_teacher_model(args, model_provider)
    # Print setup timing.
    log.info("done with setup ...")
    timers.log(
        ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
        barrier=True,
    )
    if not args.skip_train:
        log.info("training ...")
        if args.dataloader_type == "cyclic" and args.retro_add_retriever:
            args.train_iters = args.retro_cyclic_train_iters
            log.info("retro cyclic train iters : %d" % args.train_iters)
        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
            )
        print_datetime("after training is done")
        # Clean the model
        if args.compression_training:
            model = [redundancy_clean(model[0], args.deepspeed_config_dict, mpu)]
        if args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    else:
        log.info("skipping training (--skip-train is on) ...")
        iteration = args.iteration
    config = core_transformer_config_from_args(args)
    if args.do_valid:
        prefix = f"iteration {iteration} on {args.eval_iters * args.global_batch_size}-sample draw from validation set"
        _ = evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )
    if args.do_test:
        prefix = f"iteration {iteration} on {args.eval_iters * args.global_batch_size}-sample draw from test set"
        _ = evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            test=True,
        )
    return model


@dlp.log
def update_train_iters(args):
    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // args.global_batch_size
        args.train_iters = iterations

    log.info("setting training iterations to {}".format(args.train_iters))


@dlp.log
def setup_teacher_model(args, model_provider):
    log.info("***>>>>> Student model checkpoint iteration:{}".format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    log.info("***>>>>> Setting up the teacher model")

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    log.info("***>>>>> Teacher model:{}".format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def get_model(
    model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True
):
    """Build the model."""
    args = get_args()
    accelerator = get_accelerator()
    assert accelerator is not None
    assert args is not None
    args.model_type = model_type

    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert (
                    args.pipeline_model_parallel_split_rank is not None
                ), "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert (
        args.allow_transformer_engine or args.transformer_impl == "local"
    ), "Transformer Engine is only approved for GPT models"

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {})={}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum(
                            [
                                p.ds_numel if hasattr(p, "ds_id") else p.nelement()
                                for p in model_module.parameters()
                            ]
                        )
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.to(DEVICE_TYPE)

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == "torch":
            i = accelerator.current_device()
            model = [
                torchDDP(
                    model_module,
                    device_ids=[i],
                    output_device=i,
                    process_group=mpu.get_data_parallel_group(),
                )
                for model_module in model
            ]

        elif args.DDP_impl == "local":
            model = [
                LocalDDP(
                    model_module,
                    args.accumulate_allreduce_grads_in_fp32,
                    args.use_contiguous_buffers_in_local_ddp,
                )
                for model_module in model
            ]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError(
                "Unknown DDP implementation specified: "
                "{}. Exiting.".format(args.DDP_impl)
            )

    return model


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()
    assert args is not None
    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
           
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

        if args.lr_constant_fraction is not None:
            lr_constant_steps = args.lr_constant_fraction * lr_decay_steps
        else:
            lr_constant_steps = args.lr_constant_iters * args.global_batch_size
        lr_cooldown_steps = args.lr_cooldown_fraction * lr_decay_steps
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
        if args.lr_constant_fraction is not None:
            lr_constant_steps = args.lr_constant_fraction * lr_decay_steps
        else:
            lr_constant_steps = args.lr_constant_samples
        lr_cooldown_steps = args.lr_cooldown_fraction * lr_decay_steps
    else:
        raise Exception("either train-iters or train-samples should be provided.")

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        constant_lr=args.constant_lr,
        lr_constant_steps=lr_constant_steps,
        lr_cooldown_steps=lr_cooldown_steps,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
    )

    return opt_param_scheduler


@dlp.log
def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    assert args is not None
    log.info("***>>>>> Args:{}".format(args))
    model = get_model(model_provider_func)
    optimizer = None
    lr_scheduler = None
    if args.deepspeed:
        # When loading just the model weights, ZeRO can be disabled.
        if "zero_optimization" in args.deepspeed_config_dict:
            del args.deepspeed_config_dict["zero_optimization"]

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0], config=args.deepspeed_config_dict
        )

        assert not isinstance(model, deepspeed.PipelineEngine), (
            "Weight loading only mode is not supported in " "pipeline parallelism yet."
        )
        model = [model]
    print_datetime("before load checkpoint")
    if args.load is not None:
        _ = load_checkpoint(
            model, optimizer, lr_scheduler, strict=True, load_only_weights=True
        )
    print_datetime("after load checkpoint weights")
    return model, optimizer, lr_scheduler


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    no_wd_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
    teacher=False,
    data_post_process=None,
    build_train_valid_test_datasets_provider=None,
):
    """Setup model and optimizer."""
    args = get_args()
    assert args is not None
    model = get_model(model_provider_func, model_type)
    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        log.info("***>>>>> Student model, global step:{}".format(student_global_steps))
    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config_dict, mpu)]
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    if args.inference:
        optimizer = None
        opt_param_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:

            optimizer = get_megatron_optimizer(
                model, no_wd_decay_cond, scale_lr_cond, lr_mult
                )
        # opt_param_scheduler is the old lr_scheduler plus weight decay scheduling
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    if args.deepspeed:
        log.info("DeepSpeed is enabled.")
        # pp = mpu.get_pipeline_model_parallel_world_size()
        if (
            args.data_efficiency_curriculum_learning
            and build_train_valid_test_datasets_provider is not None
        ):
            log.info(
                "Caught 'args.data_efficiency_curriculum_learning' "
                "and 'build_train_valid_test_datasets_provider is not None'"
            )
            train_ds = None
            # Only need to build dataset on tp rank 0 since Megatron has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if mpu.get_tensor_model_parallel_rank() == 0:
                log.info("Caught 'mpu.get_tensor_model_parallel_rank() == 0'")
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                    update_train_iters(args)
                else:
                    train_samples = args.train_iters * args.global_batch_size
                log.info(f"{train_samples=}")
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (
                    args.train_iters // args.eval_interval + 1
                ) * args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [
                    train_samples,
                    eval_iters * args.global_batch_size,
                    test_iters * args.global_batch_size,
                ]
                log.info(f"{train_val_test_num_samples=}")
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples
                )
            with Profile("deepspeed.initialize"):
                model, optimizer, args.deepspeed_dataloader, opt_param_scheduler = (
                    deepspeed.initialize(
                        model=model[0],
                        optimizer=optimizer,
                        args=args,
                        lr_scheduler=opt_param_scheduler,
                        training_data=train_ds,
                        mpu=mpu if args.no_pipeline_parallel else None,
                        config=args.deepspeed_config_dict,
                    )
                )
            model.set_data_post_process_func(data_post_process)
        else:
            log.info(
                "Did NOT catch: ('args.data_efficiency_curriculum_learning' "
                "and 'build_train_valid_test_datasets_provider is not None')"
            )
            tds0 = time.time()
            if os.environ.get("PYINSTRUMENT_PROFILER", None):
                profiler = ez.profile.get_context_manager(rank=RANK, outdir=args.save)
            else:
                profiler = Profile("deepspeed.initialize")
            log.info("Calling 'deepspeed.initialize'...")
            log.info(f"Wrapped with: {profiler=}")
            with profiler:
                model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                    model=model[0],
                    optimizer=optimizer,
                    args=args,
                    lr_scheduler=opt_param_scheduler,
                    mpu=mpu if args.no_pipeline_parallel else None,
                    config=args.deepspeed_config_dict,
                )
            log.info(f"'deepspeed.initialize' took: {time.time() - tds0:.5f}s")
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)
            assert (
                model.grid.get_pipe_parallel_rank()
                == mpu.get_pipeline_model_parallel_rank()
            )
            assert (
                model.grid.get_slice_parallel_rank()
                == mpu.get_tensor_model_parallel_rank()
            )
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]
    # Compression has its own checkpoint loading path (e.g, loading both teacher
    # and student models). So if compression is enabled, we skip the following
    # checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        if args.load is not None:
            timers = get_timers()
            assert timers is not None
            timers("load-checkpoint", log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
            timers("load-checkpoint").stop(barrier=True)
            timers.log(["load-checkpoint"])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps
    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == "local"
    # get model without FP16 and/or TorchDDP wrappers
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
    ):
        log.info("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            assert optimizer is not None
            optimizer.reload_model_params()
    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)
    return model, optimizer, opt_param_scheduler


@dlp.log
def train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config
):
    """Single training step."""
    args = get_args()
    timers = get_timers()
    accelerator = get_accelerator()
    assert args is not None and timers is not None and accelerator is not None
    grad_norm = None
    num_zeros_in_grad = None
    if args.deepspeed and args.ds_pipeline_enabled:
        num_zeros_in_grad = 0
        assert isinstance(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        additional_losses = model[0].get_additional_losses()
        loss_key = (
            "lm loss" if additional_losses is None else "loss"
        )  # use "lm loss" for backward compatibility
        loss_dict = OrderedDict({loss_key: loss})
        if additional_losses is not None:
            loss_dict.update(additional_losses)
        grad_norm = model[0].get_global_grad_norm()
        update_successful = model[0].was_step_applied()
        skipped_iter = 0 if update_successful else 1
        return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == "local" and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        optimizer.zero_grad()

    # Forward pass.
    timers("forward-backward", log_level=1).start(barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()
    if args.mos or args.kd:
        # args.teacher_forward is used as global variable to enable kd loss
        # calculation in forward pass. Users do not need to set it in the
        # command line to use kd.
        args.teacher_forward = True

    # set timers to None if none of the timers in fwd_bwd are active, just to save the checks
    if args.timing_log_level < 2:
        config.timers = None

    num_microbatches = get_num_microbatches()
    assert num_microbatches is not None
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers("forward-backward").stop()
    if args.mos or args.kd:
        args.teacher_forward = False

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1 and accelerator is not None:
        accelerator.empty_cache()

    # Reduce gradients.
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    if args.deepspeed:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        try:
            model[0].step(lr_kwargs={"increment": increment})
            update_successful = model[0].was_step_applied()
        except Exception:
            update_successful = False
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers("optimizer").stop()

    # Gather params.
    if not args.deepspeed and update_successful:
        optimizer.gather_model_params(args, timers)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0 if update_successful else 1
        grad_norm = model[0].get_global_grad_norm()
        # Empty unused memory.
        if args.empty_unused_memory_level >= 2 and accelerator is not None:
            accelerator.empty_cache()
        # XXX: [saforem2]: ----------------------------------------------------
        # Is `num_zeros_in_grad` worth calculating (/ implementing) ??
        # the `Megatron`-specific implementation is at:
        # [megatron.optimizer.clip_grads.count_zeros_fp32](./optimizer/clip_grads.py)
        # For now, explicitly set to None
        # ---------------------------------------------------------------------
        num_zeros_in_grad = None
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1
    
    # Empty unused memory.
    if args.empty_unused_memory_level >= 2 and accelerator is not None:
        accelerator.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    assert timers is not None
    # Extra barrier is added to make sure
    # all ranks report the max time.
    # assert timers is not None
    timers("save-checkpoint", log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers("save-checkpoint").stop(barrier=True)
    checkpoint_throughput_calculator(
        model, timers("save-checkpoint").elapsed(reset=False)
    )
    timers.log(["save-checkpoint"])


@dlp.log
def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    accelerator = get_accelerator()
    assert args is not None and timers is not None and accelerator is not None
    # Write args to tensorboard
    write_args_to_tensorboard()
    assert accelerator is not None
    setup_profiler(args, accelerator.device_name())
    
    if args.random_ltd:
        # random-ltd requires different randomness on each rank
        import random

        random.seed(args.seed + torch.distributed.get_rank())
    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()
    grad_norm = None
    # Tracking loss.
    total_loss_dict = {}
    loss_dict = {"skipped_iter": 0}
    # Iterations.
    iteration = args.iteration
    # Translate args to core configuration
    config = core_transformer_config_from_args(args)
    num_skipped_iters = 0
    if not args.deepspeed:
        config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    timers("interval-time", log_level=0).start(barrier=True)
    print_datetime("before the start of training step")
    report_memory_flag = True
    if args.random_ltd:
        assert model[0].random_ltd_enabled()
        args.random_ltd_layer_num = model[
            0
        ].random_ltd_scheduler.get_random_ltd_layer_num()
    ranges_to_skip = None
    if args.train_range_to_skip is not None:
        assert (
            len(args.train_range_to_skip) % 2 == 0
        ), f"""Expected --train-range-to-skip to have an even number of values.
            Received: {len(args.train_range_to_skip)}
            """
        ranges_to_skip = list(
            zip(
                args.train_range_to_skip[::2],
                args.train_range_to_skip[1::2],
            )
        )
    
    # Learning rate finder mode
    if hasattr(args, 'lr_finder') and args.lr_finder:
        # Calculate number of iterations to use (10% of train_iters)
        finder_iters = max(1, int(args.train_iters * 0.1))
        
        # Initialize tracking variables for LR finder
        lr_finder_losses = []
        lr_finder_lrs = []
        
        # Initialize loss smoothing variables
        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0
        beta = 0.98  # Smoothing factor
        
        # Set initial learning rate and calculate multiplier
        init_lr = 1e-6
        max_lr = 1.0
        mult = (max_lr / init_lr) ** (1 / finder_iters)
        
        # Set initial learning rate
        curr_lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        
        # Turn on training mode which enables dropout
        for model_module in model:
            model_module.train()
            
        # Get configuration for training
        config = core_transformer_config_from_args(args)
        if not args.deepspeed:
            config.grad_scale_func = optimizer.scale_loss
        config.timers = timers
        
        log.info(f"Running learning rate finder for {finder_iters} iterations (10% of {args.train_iters})")
        
        # Main LR finder loop
        for i in range(finder_iters):
            # Execute training step
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
                forward_step_func,
                train_data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
            )
            
            # Skip iterations that didn't update weights
            if skipped_iter:
                continue
            
            # Get loss value (use first available loss if multiple present)
            if 'lm loss' in loss_dict:
                loss_val = loss_dict['lm loss']
            else:
                loss_val = next(iter(loss_dict.values()))
            
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            
            # Update batch counter
            batch_num += 1
            
            # Compute smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss_val
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                log.info(f"Loss exploding at lr={curr_lr:.8f}, stopping LR finder")
                break
                
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            
            # Record values for plotting
            lr_finder_losses.append(smoothed_loss)
            lr_finder_lrs.append(curr_lr)
            
            # Log to wandb
#            if wandb is not None and wandb.run is not None:
  #              wandb.log({
 #               "lr_finder/learning_rate": curr_lr,
 #               "lr_finder/raw_loss": loss_val,
 #               "lr_finder/smoothed_loss": smoothed_loss,
 #               "lr_finder/iteration": i,
 #               "lr_finder/log_lr": math.log10(curr_lr)
 #               })


            # Print progress
            if (i+1) % args.log_interval == 0:
                log.info(f"LR Finder: iteration {i+1}/{finder_iters}, "
                         f"lr: {curr_lr:.8f}, loss: {smoothed_loss:.4f}")
            
            # Update the learning rate for the next step (bypassing scheduler)
            curr_lr *= mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
                
        # Save raw data (on rank 0 only)
        if mpu.get_data_parallel_rank() == 0:
            # Create the results directory if it doesn't exist
            os.makedirs(f'{args.save}/lr_finder', exist_ok=True)
            
            # Save raw data to a simple CSV file
            import csv
            with open(f'{args.save}/lr_finder/lr_finder_data.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['learning_rate', 'loss'])
                for lr, loss in zip(lr_finder_lrs, lr_finder_losses):
                    writer.writerow([lr, loss])
            
            # Also save as numpy arrays for convenience
            try:
                import numpy as np
                np.savez(
                    f'{args.save}/lr_finder/lr_finder_data.npz',
                    learning_rates=np.array(lr_finder_lrs),
                    losses=np.array(lr_finder_losses)
                )
            except ImportError:
                pass
                
            log.info(f"LR finder completed. Results saved to {args.save}/lr_finder/")
        
        # Return after LR finder is done
        return args.iteration    

    while iteration < args.train_iters and (
        args.train_tokens is None or args.consumed_train_tokens < args.train_tokens
    ):
        trigger(on_step_begin)
        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
            model[0].set_train_batch_size(global_batch_size)
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            curriculum_seqlen = args.curriculum_scheduler.update_difficulty(
                args.iteration + 1
            )
            if iteration == 0 or curriculum_seqlen != args.curriculum_seqlen:
                if args.use_rotary_position_embeddings:
                    update_rotary_pos_emb(curriculum_seqlen)
            args.curriculum_seqlen = curriculum_seqlen
        args.curr_iteration = iteration
        if ranges_to_skip is not None and any(
            [i <= (iteration + 1) <= j for (i, j) in ranges_to_skip]
        ):
            log.info(f"Caught {iteration + 1} in 'ranges_to_skip', skipping!")
            skipped_iter = 1
            num_skipped_iters += 1
            num_zeros_in_grad = None
            gas = args.deepspeed_config_dict["gradient_accumulation_steps"]
            for microstep in range(gas):
                _batch = next(train_data_iterator)
                _tokens = _batch["text"]
                if (
                    iteration < 10
                    and os.environ.get("DUMP_SKIPPED_ITERS", None)
                    and RANK == 0
                ):
                    log.info(f"{_tokens.shape}, {len(train_data_iterator)=}")
                    log.info(
                        f"{iteration=} [{microstep}/{gas}]: ({_tokens.shape})\n{_tokens[:10]=}"
                    )

            increment = (
                get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            )
            model[0].skipped_steps += 1
            model[0].global_steps += 1
            model[0].micro_steps += 1
            model[0].global_samples += model[0].train_batch_size()
            opt_param_scheduler.step(increment=increment)
        else:
            if os.getenv("TORCH_PROFILER_ENABLE") == "2":
                from torch.profiler import profile, ProfilerActivity

                try:
                    activities = [
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA,
                        ProfilerActivity.XPU,  # type:ignore
                    ]
                except Exception:
                    log.warning("TORCH PROFILER WARNING: XPU is not supported")
                    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                with profile(activities=activities) as prof:
                    loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
                        forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config,
                    )
                prof.export_chrome_trace(
                    f"{args.trace_dir}/torch-trace-{RANK}-of-{WORLD_SIZE}-step{iteration}.json"
                )
            else:
                loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
                    forward_step_func,
                    train_data_iterator,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    config,
                )
        iteration += 1
        args.iteration = iteration
        

        new_samples = (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[
                0
            ].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (
                    args.actual_seq_length
                    * (args.num_layers - args.random_ltd_layer_num)
                    + args.random_ltd_reserved_length * args.random_ltd_layer_num
                ) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, "data_efficiency_curriculum_learning_numel"):
                act_mbsz = (
                    args.data_efficiency_curriculum_learning_numel
                    / args.curriculum_seqlen
                )
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += (
                    mpu.get_data_parallel_world_size()
                    * get_num_microbatches()
                    * act_token
                )
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length
        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, "cur_scale"):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        
        ### Begin MuP Code ###
        if (args.enable_mup or args.enable_depth_scale) and args.mup_coord_check:
            if iteration <=20:
                mup_coord_check(model)
        ### End MuP Code ###

        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            optimizer.param_groups[0]["lr"],
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            model,
            optimizer,
        )
        # Autoresume
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(
                iteration, model, optimizer, opt_param_scheduler
            )
        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                False,
            )
        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            # if any(signal_handler.signals_received()):
            if signal_handler is not None and any(signal_handler.signals_received()):
                save_checkpoint_and_time(
                    iteration, model, optimizer, opt_param_scheduler
                )
                print_datetime("exiting program after receiving SIGTERM.")
                sys.exit()
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler)
            saved_checkpoint = True
        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = accelerator.IntTensor([train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(
                        iteration, model, optimizer, opt_param_scheduler
                    )
                print_datetime("exiting program after {} minutes".format(train_time))
                sys.exit()
        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration, model, optimizer, opt_param_scheduler
                )
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            sys.exit()
        trigger(on_step_end)
        # Exiting based on kill switch file
        if found_kill_switch():
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration, model, optimizer, opt_param_scheduler
                )
            torch.distributed.barrier()
            print_datetime(
                f"Detected kill switch at {args.kill_switch_file}, "
                f"iteration={iteration}. Exiting"
            )
            sys.exit()
    return iteration


@dlp.log
def evaluate(
    forward_step_func,
    data_iterator,
    model,
    process_non_loss_data_func,
    config,
    verbose=False,
):
    """Evaluation."""
    args = get_args()
    accelerator = get_accelerator()
    assert args is not None and accelerator is not None
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    total_loss_dict = {}

    num_microbatches = get_num_microbatches()
    assert num_microbatches is not None
    forward_backward_func = get_forward_backward_func()

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                log.info("Evaluating iter {}/{}".format(iteration, args.eval_iters))

            # Don't care about timing during evaluation
            config.timers = None
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{"lm loss": loss}] * num_microbatches
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                )
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                accelerator.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if "moe" not in key:
                            total_loss_dict[key] = (
                                total_loss_dict.get(key, accelerator.FloatTensor([0.0]))
                                + loss_dict[key]
                            )

            args.consumed_valid_samples += (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * num_microbatches
            )
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * num_microbatches

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty(
            args.iteration + 1
        )
        if args.curriculum_seqlen < args.seq_length:
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    return total_loss_dict, collected_non_loss_data


@dlp.log
def evaluate_and_print_results(
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    process_non_loss_data_func,
    config,
    verbose=False,
    write_to_tensorboard=True,
    test=False,
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    assert args is not None
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
    )
    key = "test" if test else "val"
    if wandb is not None and wandb.run is not None:
        wandb.log({
            f"{key}/iteration": iteration,
            **{f"{key}/{k}": v for k, v in total_loss_dict.items()},
            **{
                f"{key}/ppl_{k}": math.exp(min(20, v.item()))
                for k, v in total_loss_dict.items()
            },
        })
    string = " validation loss at {} | ".format(prefix)
    for key in total_loss_dict:
        string += f"{key} value={total_loss_dict[key].item():.6f}"
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += f"{key} PPL={ppl:.6f}"
        # string += '{} PPL={:.6f} | '.format(key, ppl)
        if writer is not None and is_last_rank():
            data_type = "test" if test else "validation"
            writer.add_scalar(
                f"lm-loss-validation/{key} {data_type}",
                total_loss_dict[key].item(),
                iteration,
            )
            writer.add_scalar(
                f"lm-loss-validation/{key} {data_type} vs samples",
                total_loss_dict[key].item(),
                args.consumed_train_samples,
            )
            writer.add_scalar(
                f"lm-loss-validation/{key} {data_type} vs tokens",
                total_loss_dict[key].item(),
                args.consumed_train_tokens,
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(
                    f"lm-loss-validation/{key} {data_type} ppl", ppl, iteration
                )
                writer.add_scalar(
                    f"lm-loss-validation/{key} {data_type} ppl vs samples",
                    ppl,
                    args.consumed_train_samples,
                )
                writer.add_scalar(
                    f"lm-loss-validation/{key} {data_type} ppl vs tokens",
                    ppl,
                    args.consumed_train_tokens,
                )

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    log.info("-" * length)
    log.info(string)
    log.info("-" * length)
    return total_loss_dict


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    assert args is not None
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    ]
    log.info(" > datasets target sizes (minimum size):")
    log.info("    train:      {}".format(train_val_test_num_samples[0]))
    log.info("    validation: {}".format(train_val_test_num_samples[1]))
    log.info("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""
    args = get_args()
    accelerator = get_accelerator()
    assert args is not None and accelerator is not None
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
    log.info("> building train, validation, and test datasets ...")
    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert (
            args.train_samples is None
        ), "only backward compatiblity support for iteration-based training"
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (
                (args.iteration // args.eval_interval)
                * args.eval_iters
                * args.global_batch_size
            )
    # Data loader only on rank 0 of each model parallel group.
    ds_sequence_parallel = (
        mpu.get_sequence_parallel_world_size() > 1 or args.force_ds_sequence_parallel
    )
    rank_in_parallel_group = (
        mpu.get_sequence_parallel_rank()
        if ds_sequence_parallel
        else mpu.get_tensor_model_parallel_rank()
    )
    if rank_in_parallel_group == 0:
        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider
        )
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples
        )
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples
        )
        test_dataloader = build_pretraining_data_loader(test_ds, 0)
        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = accelerator.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = accelerator.LongTensor([0, 0, 0])
    # Broadcast num tokens.
    if ds_sequence_parallel:
        torch.distributed.broadcast(
            flags,
            mpu.get_sequence_parallel_src_rank(),
            group=mpu.get_sequence_parallel_group(),
        )
    else:
        torch.distributed.broadcast(
            flags,
            mpu.get_tensor_model_parallel_src_rank(),
            group=mpu.get_tensor_model_parallel_group(),
        )
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()
    return train_dataloader, valid_dataloader, test_dataloader


@dlp.log
@ez.dist.timeitlogit(rank=RANK)
def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()
    assert args is not None

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = (
        build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)
    )

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic"]

    if train_dataloader is not None:
        train_data_iterator = (
            iter(train_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = (
            iter(valid_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(valid_dataloader))
        )
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = (
            iter(test_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(test_dataloader))
        )
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator

# 🧊 Cooling Down AuroraGPT-2B Checkpoints

## 📉 Simple Experiment to Compare Validation Loss

![Cool Down Comparison](./assets/cooldown-grid.png)

See W&B Report: [Cooling Down
Checkpoints](https://api.wandb.ai/links/aurora_gpt/dek99dmd) for more details.

## ☃️ Cooling Down

- 256 Nodes of Aurora:

  - Cooled down over last 10\%:
    - W&B Run: [volcanic-blaze-4312](https://wandb.ai/aurora_gpt/AuroraGPT/runs/7bjj8vgu/overview?nw=nwuserforemans)

  - Explicit command:

    ```bash
    LR_DECAY_STYLE=constant \
        OPT=ipex.fusedlamb \
        OVERRIDE_CKPT_OPT_PARAM=1 \
        TRAIN_ITERS=137650 \
        GRAD_ACC_STEPS=2 \
        LOAD=test_rollback \
        DATA_FILE_LIST=ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
        bash train_alcf.sh \
            --override-opt_param-scheduler \
            --min-lr=2e-5 \
            --lr_constant_plus_cooldown \
            --lr_constant_plus_cooldown_frac=0.9
    ```

  - <details closed><summary>Example:</summary>

    ```bash
    #[🐍 aurora_frameworks-2025.2.0](👻 Megatron-DeepSpeed-aurora_frameworks-2025.2.0)
    #[/f/A/A/E/A/c/r/Megatron-DeepSpeed][🌱 main][✅] [⏱️ 26m13s]
    #[11/10/25 @ 10:19:03][x4417c6s4b0n0]
    ; LR_DECAY_STYLE=constant \
        LR=0.0002 \
        OPT=ipex.fusedlamb \
        OVERRIDE_CKPT_OPT_PARAM=1 \
        TRAIN_ITERS=137650 \
        GRAD_ACC_STEPS=2 \
        LOAD=test_rollback \
        DATA_FILE_LIST=ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
        bash train_alcf.sh \
            --lr_constant_plus_cooldown \
            --lr_constant_plus_cooldown_frac=0.9 \
            --min-lr=2e-5 \
            --override-opt_param-scheduler
    [2025-11-10-095114][I][] Detected PBS scheduler environment.
    [2025-11-10-095114][I][] running [ezpz_setup_env]...
    [2025-11-10-095114][I][] [PYTHON]
    [2025-11-10-095114][I][]   - Found both conda_prefix and virtual_env in environment.
    [2025-11-10-095114][I][]   - Using conda from: /opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0
    [2025-11-10-095114][I][]   - Using venv from: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/cooldown-experiments/run-pt25-ipex-fusedlamb-
    256-nodes/Megatron-DeepSpeed/venvs/aurora/Megatron-DeepSpeed-aurora_frameworks-2025.2.0
    [2025-11-10-095114][I][]   - Using python from: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/cooldown-experiments/run-pt25-ipex-fusedlam
    b-256-nodes/Megatron-DeepSpeed/venvs/aurora/Megatron-DeepSpeed-aurora_frameworks-2025.2.0/bin/python3
    [2025-11-10-095114][I][] [JOB]
    [2025-11-10-095114][I][]   - Setting up env for foremans
    [2025-11-10-095114][I][]   - Detected pbs scheduler
    [2025-11-10-095114][I][]   - Machine: aurora
    [2025-11-10-095114][I][]   - Hostname: x4417c6s4b0n0
    [2025-11-10-095116][I][]   - PBS_JOBID=8140578.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        to calculate:
          - num_hosts: 256
          - num_cores_per_host: 208
          - num_cpus_per_host: 104
          - num_gpus_per_host: 12
          - depth: 8
          - num_gpus: 3072
    [2025-11-10-095116][I][] [HOSTS] - ezpz_print_hosts
    [2025-11-10-095116][I][]   - Detected PBS Scheduler
    [2025-11-10-095116][I][] [HOSTS]
    [2025-11-10-095116][I][]   - HOSTFILE=/var/spool/pbs/aux/8140578.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2025-11-10-095116][I][]   - NHOSTS=256
    [2025-11-10-095116][I][]   - HOSTS:
    [2025-11-10-095116][I][]     - [host:0] - x4417c6s4b0n0.hsn.cm.aurora.alcf.anl.gov
    # [...clipped...]
    [2025-11-10 09:56:45,204] [INFO] [config.py:684:__init__] Config mesh_device None world_size = 3072
    [2025-11-10 09:56:45,302] [INFO] [utils.py:781:see_memory_usage] Before Building Model
    [2025-11-10 09:56:45,303] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 0.0 GB         Max_CA 0 GB
    [2025-11-10 09:56:45,303] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 73.83 GB, percent = 6.5%
    [2025-11-10 09:56:45,304] [INFO] [config.py:684:__init__] Config mesh_device None world_size = 3072
    [2025-11-10 09:56:45,320815][I][Megatron-DeepSpeed/pretrain_gpt_alcf:151:model_provider] --------------------------------------------------------------------------------
    [2025-11-10 09:56:45,321819][I][Megatron-DeepSpeed/pretrain_gpt_alcf:152:model_provider] Number of parameters in model: 1986054144
    [2025-11-10 09:56:45,322546][I][Megatron-DeepSpeed/pretrain_gpt_alcf:153:model_provider] --------------------------------------------------------------------------------
    [2025-11-10 09:56:45,484] [INFO] [utils.py:781:see_memory_usage] After Building Model
    [2025-11-10 09:56:45,485] [INFO] [utils.py:782:see_memory_usage] MA 3.71 GB         Max_MA 3.71 GB         CA 3.72 GB         Max_CA 4 GB
    [2025-11-10 09:56:45,485] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 73.84 GB, percent = 6.5%
    [2025-11-10 09:56:45,485921][I][Megatron-DeepSpeed/pretrain_gpt_alcf:161:model_provider] Patching tensorboard from checkpoints/AuroraGPT-2B-ws3072-ds-stage1-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optipex.fusedlamb-lr0.0002-lwf0.05_ntok0B_flash/tensorboard
    wandb: WARNING The get_url method is deprecated and will be removed in a future release. Please use `run.url` instead.
    [2025-11-10 09:56:51,975268][I][Megatron-DeepSpeed/pretrain_gpt_alcf:168:model_provider] Updating WandB run.config: [sandy-darkness-4309](https://wandb.ai/aurora_gpt/AuroraGPT/runs/2cmpsosr)
     > number of parameters on (tensor, pipeline) model parallel rank (0, 0)=1986054144
    [2025-11-10 09:56:51,979967][I][megatron/optimizer_param_scheduler:89:__init__] > learning rate decay style: constant
    [2025-11-10 09:56:51,980855][I][megatron/training:725:setup_model_and_optimizer] DeepSpeed is enabled.
    [2025-11-10 09:56:51,981610][I][megatron/training:780:setup_model_and_optimizer] Did NOT catch: ('args.data_efficiency_curriculum_learning' and 'build_train_valid_test_datasets_provider is not None')
    [2025-11-10 09:56:51,982375][I][megatron/training:789:setup_model_and_optimizer] Calling 'deepspeed.initialize'...
    [2025-11-10 09:56:51,983047][I][megatron/training:790:setup_model_and_optimizer] Wrapped with: profiler=<megatron.utils.Profile object at 0x1472dc6c21a0>
    [2025-11-10 09:56:51,983] [INFO] [logging.py:107:log_dist] [Rank 0] DeepSpeed info: version=0.17.5, git-hash=unknown, git-branch=unknown
    # [...clipped...]
    [2025-11-10 09:43:36,149] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 715.45 | bwd_microstep: 2402.91 | bwd_inner_microstep: 1223.71 | bwd_allreduce_microstep: 1178.40 | step_microstep: 160.78
    [2025-11-10 09:43:36,149] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 715.40 | bwd: 2402.86 | bwd_inner: 1223.70 | bwd_allreduce: 1178.40 | step: 160.79
    [2025-11-10 09:43:36,161992][I][megatron/training_log:402:training_log]  iteration=  136004/  136100 | consumed_samples=   835608576 | consumed_tokens=6845305454592 | elapsed_time_per_iteration_ms=3415.3 | learning_rate=2.0127e-05 | global_batch_size= 6144 | lm loss=12.316531 | loss_scale=1.0 | grad_norm=1020.352 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1798.971 | tokens_per_gpu_per_second_tgs=4797.255 | TFLOPs=53.66 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (3198.27, 3199.53)
        optimizer ......................................: (158.53, 166.36)
    [2025-11-10 09:43:39,572] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 167.02 | optimizer_gradients: 0.46 | optimizer_step: 1.13
    [2025-11-10 09:43:39,573] [INFO] [logging.py:107:log_dist] [Rank 0] step=136005, skipped=0, lr=[2.012568679065123e-05, 2.012568679065123e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:43:39,573] [INFO] [timer.py:264:stop] epoch=0/micro_step=5/global_step=5, RunningAvgSamplesPerSec=2443.9516881034883, CurrSamplesPerSec=2553.443452275532, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:43:39,574] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 713.96 | bwd_microstep: 2398.60 | bwd_inner_microstep: 1222.63 | bwd_allreduce_microstep: 1175.22 | step_microstep: 173.52
    [2025-11-10 09:43:39,574] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 713.90 | bwd: 2398.57 | bwd_inner: 1222.62 | bwd_allreduce: 1175.27 | step: 173.53
    [2025-11-10 09:43:39,582716][I][megatron/training_log:402:training_log]  iteration=  136005/  136100 | consumed_samples=   835614720 | consumed_tokens=6845355786240 | elapsed_time_per_iteration_ms=3420.2 | learning_rate=2.01257e-05 | global_batch_size= 6144 | lm loss=11.967899 | loss_scale=1.0 | grad_norm=735.534 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1796.410 | tokens_per_gpu_per_second_tgs=4790.426 | TFLOPs=53.59 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (3194.05, 3195.63)
        optimizer ......................................: (170.97, 175.03)
    [2025-11-10 09:43:46,436] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 157.82 | optimizer_gradients: 0.47 | optimizer_step: 1.12
    [2025-11-10 09:43:46,436] [INFO] [logging.py:107:log_dist] [Rank 0] step=136006, skipped=0, lr=[2.0124363314442905e-05, 2.0124363314442905e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:43:46,437] [INFO] [timer.py:264:stop] epoch=0/micro_step=6/global_step=6, RunningAvgSamplesPerSec=2472.2740699287133, CurrSamplesPerSec=2561.3206693919196, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:43:46,438] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 714.84 | bwd_microstep: 2347.61 | bwd_inner_microstep: 1165.11 | bwd_allreduce_microstep: 1181.70 | step_microstep: 164.54
    [2025-11-10 09:43:46,438] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 714.77 | bwd: 2347.57 | bwd_inner: 1165.11 | bwd_allreduce: 1181.73 | step: 164.55
    [2025-11-10 09:43:46,446173][I][megatron/training_log:402:training_log]  iteration=  136006/  136100 | consumed_samples=   835620864 | consumed_tokens=6845406117888 | elapsed_time_per_iteration_ms=6863.2 | learning_rate=2.01244e-05 | global_batch_size= 6144 | lm loss=11.744588 | loss_scale=1.0 | grad_norm=527.276 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=895.207 | tokens_per_gpu_per_second_tgs=2387.220 | TFLOPs=26.70 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (6639.44, 6640.87)
        optimizer ......................................: (161.40, 165.76)
    [2025-11-10 09:43:49,839] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 150.38 | optimizer_gradients: 0.47 | optimizer_step: 1.12
    [2025-11-10 09:43:49,839] [INFO] [logging.py:107:log_dist] [Rank 0] step=136007, skipped=0, lr=[2.012303984797229e-05, 2.012303984797229e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:43:49,840] [INFO] [timer.py:264:stop] epoch=0/micro_step=7/global_step=7, RunningAvgSamplesPerSec=2492.441497638933, CurrSamplesPerSec=2576.511402462105, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:43:49,841] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 714.13 | bwd_microstep: 2397.86 | bwd_inner_microstep: 1224.18 | bwd_allreduce_microstep: 1172.96 | step_microstep: 156.83
    [2025-11-10 09:43:49,841] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 714.07 | bwd: 2397.83 | bwd_inner: 1224.17 | bwd_allreduce: 1172.99 | step: 156.84
    [2025-11-10 09:43:49,849423][I][megatron/training_log:402:training_log]  iteration=  136007/  136100 | consumed_samples=   835627008 | consumed_tokens=6845456449536 | elapsed_time_per_iteration_ms=3402.7 | learning_rate=2.0123e-05 | global_batch_size= 6144 | lm loss=11.613136 | loss_scale=1.0 | grad_norm=579.721 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1805.618 | tokens_per_gpu_per_second_tgs=4814.980 | TFLOPs=53.86 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (3191.89, 3193.07)
        optimizer ......................................: (154.73, 158.25)
    [2025-11-10 09:43:55,173] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 144.15 | optimizer_gradients: 0.47 | optimizer_step: 1.13
    [2025-11-10 09:43:55,174] [INFO] [logging.py:107:log_dist] [Rank 0] step=136008, skipped=0, lr=[2.0121716391239166e-05, 2.0121716391239166e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:43:55,175] [INFO] [timer.py:264:stop] epoch=0/micro_step=8/global_step=8, RunningAvgSamplesPerSec=2214.823701527157, CurrSamplesPerSec=1422.5676763766571, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:43:55,175] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 714.49 | bwd_microstep: 2325.32 | bwd_inner_microstep: 1170.29 | bwd_allreduce_microstep: 1154.19 | step_microstep: 150.72
    [2025-11-10 09:43:55,176] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 714.43 | bwd: 2325.28 | bwd_inner: 1170.29 | bwd_allreduce: 1154.22 | step: 150.73
    [2025-11-10 09:43:55,183901][I][megatron/training_log:402:training_log]  iteration=  136008/  136100 | consumed_samples=   835633152 | consumed_tokens=6845506781184 | elapsed_time_per_iteration_ms=5334.1 | learning_rate=2.01217e-05 | global_batch_size= 6144 | lm loss=11.405020 | loss_scale=1.0 | grad_norm=230.690 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1151.839 | tokens_per_gpu_per_second_tgs=3071.570 | TFLOPs=34.36 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (5129.49, 5130.83)
        optimizer ......................................: (147.99, 152.14)
    [2025-11-10 09:43:58,569] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 138.00 | optimizer_gradients: 0.47 | optimizer_step: 1.12
    [2025-11-10 09:43:58,570] [INFO] [logging.py:107:log_dist] [Rank 0] step=136009, skipped=0, lr=[2.0120392944243313e-05, 2.0120392944243313e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:43:58,570] [INFO] [timer.py:264:stop] epoch=0/micro_step=9/global_step=9, RunningAvgSamplesPerSec=2261.5249756425674, CurrSamplesPerSec=2589.0806546406557, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:43:58,571] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 716.27 | bwd_microstep: 2402.82 | bwd_inner_microstep: 1226.62 | bwd_allreduce_microstep: 1175.49 | step_microstep: 144.53
    [2025-11-10 09:43:58,571] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 716.21 | bwd: 2402.79 | bwd_inner: 1226.63 | bwd_allreduce: 1175.53 | step: 144.54
    [2025-11-10 09:43:58,579283][I][megatron/training_log:402:training_log]  iteration=  136009/  136100 | consumed_samples=   835639296 | consumed_tokens=6845557112832 | elapsed_time_per_iteration_ms=3394.8 | learning_rate=2.01204e-05 | global_batch_size= 6144 | lm loss=11.290359 | loss_scale=1.0 | grad_norm=326.170 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1809.814 | tokens_per_gpu_per_second_tgs=4826.171 | TFLOPs=53.99 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (3194.54, 3195.97)
        optimizer ......................................: (141.84, 145.75)
    [2025-11-10 09:44:01,970] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 137.27 | optimizer_gradients: 0.48 | optimizer_step: 1.13
    [2025-11-10 09:44:01,971] [INFO] [logging.py:107:log_dist] [Rank 0] step=136010, skipped=0, lr=[2.011906950698453e-05, 2.011906950698453e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:44:01,972] [INFO] [timer.py:264:stop] epoch=0/micro_step=10/global_step=10, RunningAvgSamplesPerSec=2296.9740702454246, CurrSamplesPerSec=2580.0686696658236, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:44:01,972] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 713.33 | bwd_microstep: 2403.22 | bwd_inner_microstep: 1220.85 | bwd_allreduce_microstep: 1181.64 | step_microstep: 143.81
    [2025-11-10 09:44:01,973] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 713.27 | bwd: 2403.19 | bwd_inner: 1220.86 | bwd_allreduce: 1181.69 | step: 143.82
    [2025-11-10 09:44:01,981208][I][megatron/training_log:402:training_log]  iteration=  136010/  136100 | consumed_samples=   835645440 | consumed_tokens=6845607444480 | elapsed_time_per_iteration_ms=3401.5 | learning_rate=2.01191e-05 | global_batch_size= 6144 | lm loss=11.142616 | loss_scale=1.0 | grad_norm=184.646 | actual_seqlen= 8192 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=1806.288 | tokens_per_gpu_per_second_tgs=4816.769 | TFLOPs=53.88 |
    (min, max) time across ranks (ms):
        forward-backward ...............................: (3196.98, 3200.27)
        optimizer ......................................: (141.42, 145.17)
    [2025-11-10 09:44:05,471] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | optimizer_allgather: 131.87 | optimizer_gradients: 0.47 | optimizer_step: 1.14
    [2025-11-10 09:44:05,472] [INFO] [logging.py:107:log_dist] [Rank 0] step=136011, skipped=0, lr=[2.0117746079462598e-05, 2.0117746079462598e-05], mom=[(0.9, 0.95), (0.9, 0.95)]
    [2025-11-10 09:44:05,473] [INFO] [timer.py:264:stop] epoch=0/micro_step=11/global_step=11, RunningAvgSamplesPerSec=2327.316936098493, CurrSamplesPerSec=2602.3285686919153, MemAllocated=3.71GB, MaxMemAllocated=39.0GB
    [2025-11-10 09:44:05,473] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd_microstep: 719.44 | bwd_microstep: 2365.65 | bwd_inner_microstep: 1195.86 | bwd_allreduce_microstep: 1169.15 | step_microstep: 138.29
    [2025-11-10 09:44:05,474] [INFO] [logging.py:107:log_dist] [Rank 0] time (ms) | fwd: 719.38 | bwd: 2365.62 | bwd_inner: 1195.82 | bwd_allreduce: 1169.19 | step: 138.30
    [2025-11-10 12:29:56,532794][I][megatron/training:1534:evaluate] Evaluating iter 19/20
    [2025-11-10 12:29:57,331870][I][megatron/training:1534:evaluate] Evaluating iter 20/20
    [2025-11-10 12:29:58,133928][I][megatron/training:1692:evaluate_and_print_results] -----------------------------------------------------------------------------------------------------------------------------
    [2025-11-10 12:29:58,134910][I][megatron/training:1693:evaluate_and_print_results]  validation loss at iteration 137650 on 122880-sample draw from validation set | lm loss value=2.697780lm loss PPL=14.846734
    [2025-11-10 12:29:58,136014][I][megatron/training:1694:evaluate_and_print_results] -----------------------------------------------------------------------------------------------------------------------------
    Comm. Op            Message Size        Count               Total Latency(ms)   Avg Latency(ms)     tput_avg (Gbps)     busbw_avg (Gbps)
    broadcast
                        4.0 KB              25                  980.95              0.33                0.11                0.11
                        8.0 MB              12                  214.59              12.05               8.72                8.72
                        12.0 MB             12                  354.42              12.34               10.63               10.63
                        43.0 MB             12                  346.20              23.60               16.63               16.63
                        86.0 MB             12                  492.66              37.23               20.63               20.63
                        1000.0 MB           2                   62863.52            31431.76            11.92               11.92
    all_reduce
                        4.0 B               3300                1828.19             0.53                0.00                0.00
                        20.0 B              68                  238.60              3.36                0.00                0.00
                        100.0 KB            1650                108812.85           26.15               0.53                0.53
                        45.6 MB             1650                59143.33            31.96               24.15               24.14
                        45.62 MB            1650                72884.32            42.17               18.16               18.15
                        874.0 MB            1650                -259904.95          246.25              59.55               59.53
                        914.0 MB            1650                -296460.05          230.41              66.55               66.53
                        954.38 MB           1650                -223315.44          268.82              59.56               59.54
                        954.4 MB            1650                497077.13           285.12              56.16               56.14
    all_gather_into_tensor
                        36.0 B              1650                61948.81            37.49               0.02                0.02
                        1.23 MB             1650                -527532.78          95.50               332.86              332.75
    barrier
                        0B                  68                  6248.29             14.28               0.00                0.00
    log_summary_barrier
                        0B                  1                   498.65              498.65              0.00                0.00
    wandb.run.name: volcanic-blaze-4312
    wandb.run.url: https://wandb.ai/aurora_gpt/AuroraGPT/runs/7bjj8vgu
    wandb: updating run metadata
    wandb: uploading config.yaml
    wandb:
    wandb: Run summary:
    wandb:                learning-rate/iteration 137650
    wandb:            learning-rate/learning-rate 2e-05
    wandb: lm-loss-training/consumed_train_tokens 6928151347200
    wandb:             lm-loss-training/iteration 137650
    wandb:               lm-loss-training/lm loss 2.69441
    wandb:                         loss/grad_norm 3.39173
    wandb:                         loss/iteration 137650
    wandb:                           loss/lm loss 2.69441
    wandb:                       loss/lm loss_avg 2.69441
    wandb:                        loss/loss_scale 1
    wandb:                                    +50 ...
    wandb:
    wandb:  View run volcanic-blaze-4312 at: https://wandb.ai/aurora_gpt/AuroraGPT/runs/7bjj8vgu
    wandb:  View project at: https://wandb.ai/aurora_gpt/AuroraGPT
    wandb: Synced 8 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
    wandb: Find logs at: ./wandb/run-20251110_102334-7bjj8vgu/logs
    Application 5d196584 resources: utime=21540179s stime=2039451s maxrss=46631076KB inblock=18009722136 oublock=2015065920 minflt=16872908673 majflt=453960025 nvcsw=11096789454 nivcsw=10717725
    [2025-11-10 12:30:19,247084][I][ezpz/launch:402:launch] Execution finished with 0.
    [2025-11-10 12:30:19,247825][I][ezpz/launch:403:launch] Executing finished in 7795.27 seconds.
    [2025-11-10 12:30:19,248203][I][ezpz/launch:404:launch] Took 7795.28 seconds to run. Exiting.
    took: 2h 11m 1s
    ```

  </details>


## ♻️ Convert to Universal (Optional)

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 ALCF/ds_to_universal.py \
    --input_folder test_rollback/global_step136000 \
    --output_folder test_rollback/global_step136000_universal
```


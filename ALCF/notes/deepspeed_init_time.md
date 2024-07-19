# DeepSpeed Initialization Time on Aurora

## TODOs

- [ ] Use `ZeRO={1, 2}` @ 256 Nodes of Aurora
- [ ] Figure out bottleneck in startup time on Aurora
- [ ] Use GAS=8 on Aurora
- [ ] Weight decay too high
- [ ] Save checkpoints every ~ 1 hr
- [ ] Write weekly updates and post to GitHub

## Initialization Times

- Search for "deepspeed.initialize" in `Megatron-DeepSpeed/logs/`:

```bash
#[🌌][11:44:57 PM][foremans@aurora-uan-0010][…/Megatron-DeepSpeed/logs][🌱 alcf-startup-time][$!?]
$ rg --hidden "deepspeed\.initialize" **/**/*.log | grep took
```

### Measurements

| NUM_NODES | WORLD_SIZE |    TIME    |
|:---------:|:----------:|:----------:|
|     8     |     96     |   61.073   |
|           |            |            |
|     16    |     192    |  107.74411 |
|     16    |     192    | 107.201338 |
|     16    |     192    |  107.10853 |
|           |            |            |
|     32    |     384    |  200.23095 |
|     32    |     384    |  206.49485 |
|     32    |     384    |  200.49485 |
|           |            |            |
|     64    |     768    |  413.55765 |
|     64    |     768    |  394.92617 |
|     64    |     768    |   414.725  |
|     64    |     768    |   387.987  |
|     64    |     768    |  411.72035 |
|     64    |     768    |   394.926  |
|     64    |     768    |   409.375  |
|     64    |     768    |   393.091  |
|     64    |     768    |   412.600  |
|           |            |            |
|    128    |    1536    |  789.30077 |
|    128    |    1536    |  788.86531 |
|    128    |    1536    |  792.71864 |
|    128    |    1536    |   836.98   |
|    128    |    1536    |   801.205  |
|    128    |    1536    |   836.98   |
|    128    |    1536    |  820.9538  |
|    128    |    1536    |   707.048  |
|           |            |            |
|    256    |    3072    | 1639.62374 |
|    256    |    3072    |  1591.345  |
|    256    |    3072    | 1632.12712 |
|    256    |    3072    |  1674.444  |
|    256    |    3072    |  1618.100  |


- <details closed><summary><code>WORLD_SIZE=96</code>:</summary>

  ```bash title="deepspeed_init_times.sh"
  ws96_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-125717_96_x4420c5s5b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:59:19][INFO][training:795] - 'deepspeed.initialize' took: 61.07362s
  ws96_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-125717_96_x4420c5s5b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:59:19][INFO][training:795] - 'deepspeed.initialize' took: 61.07362s
  ws96_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-125717_96_x4420c5s5b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:59:19][INFO][training:795] - 'deepspeed.initialize' took: 61.07362s
  ```

  </details>

- <details closed><summary><code>WORLD_SIZE = 192</code>:</summary>

  ```bash
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-154948_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:52:30][INFO][training:795] - 'deepspeed.initialize' took: 107.74411s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-154948_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:52:30][INFO][training:795] - 'deepspeed.initialize' took: 107.74411s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-154948_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:52:30][INFO][training:795] - 'deepspeed.initialize' took: 107.74411s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamwschedulefree_lr0.0003_lwf0.05/20240623-163640_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:38:52][INFO][training:800] - 'deepspeed.initialize' took: 107.10853s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-160332_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:05:43][INFO][training:800] - 'deepspeed.initialize' took: 107.20138s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-160332_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:05:43][INFO][training:800] - 'deepspeed.initialize' took: 107.20138s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb6144_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-160332_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:05:43][INFO][training:800] - 'deepspeed.initialize' took: 107.20138s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamwschedulefree_lr0.0003_lwf0.05/20240623-163640_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:38:52][INFO][training:800] - 'deepspeed.initialize' took: 107.10853s
  ws192_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamwschedulefree_lr0.0003_lwf0.05/20240623-163640_192_x4716c2s6b0n0.hostmgmt2716.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:38:52][INFO][training:800] - 'deepspeed.initialize' took: 107.10853s
  ```

  </details>

- <details closed><summary><code>WORLD_SIZE = 384</code>:</summary>

  ```bash
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-164607_384_x4402c6s7b0n0.hostmgmt2402.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:52:15][INFO][training:800] - 'deepspeed.initialize' took: 206.49485s
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-164607_384_x4402c6s7b0n0.hostmgmt2402.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:52:15][INFO][training:800] - 'deepspeed.initialize' took: 206.49485s
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-164607_384_x4402c6s7b0n0.hostmgmt2402.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:52:15][INFO][training:800] - 'deepspeed.initialize' took: 206.49485s
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-223159_384_x4706c1s6b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:37:53][INFO][training:800] - 'deepspeed.initialize' took: 200.23095s
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-223159_384_x4706c1s6b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:37:53][INFO][training:800] - 'deepspeed.initialize' took: 200.23095s
  ws384_ds_stage1_nl32_hs4096_mb4_seq4096_gb12288_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-223159_384_x4706c1s6b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:37:53][INFO][training:800] - 'deepspeed.initialize' took: 200.23095s
  ```

  </details>

- <details closed><summary><code>WORLD_SIZE=768</code>:</summary>

  ```bash
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-180052_768_x4704c4s1b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 18:12:43][INFO][training:800] - 'deepspeed.initialize' took: 394.92617s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-185626_768_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 19:05:45][INFO][training:800] - 'deepspeed.initialize' took: 414.72580s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-185626_768_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 19:05:45][INFO][training:800] - 'deepspeed.initialize' took: 414.72580s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-233045_768_x4711c0s1b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 23:39:19][INFO][training:797] - 'deepspeed.initialize' took: 387.98744s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-233045_768_x4711c0s1b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 23:39:19][INFO][training:797] - 'deepspeed.initialize' took: 387.98744s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-233045_768_x4711c0s1b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 23:39:19][INFO][training:797] - 'deepspeed.initialize' took: 387.98744s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-141802_768_x4706c2s0b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:27:50][INFO][training:795] - 'deepspeed.initialize' took: 411.72035s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-141802_768_x4706c2s0b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:27:50][INFO][training:795] - 'deepspeed.initialize' took: 411.72035s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-180052_768_x4704c4s1b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 18:12:43][INFO][training:800] - 'deepspeed.initialize' took: 394.92617s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-180052_768_x4704c4s1b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 18:12:43][INFO][training:800] - 'deepspeed.initialize' took: 394.92617s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-134324_768_x4705c2s1b0n0.hostmgmt2705.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:51:19][INFO][training:795] - 'deepspeed.initialize' took: 393.09134s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-185626_768_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 19:05:45][INFO][training:800] - 'deepspeed.initialize' took: 414.72580s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-165713_768_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:06:47][INFO][training:800] - 'deepspeed.initialize' took: 389.15768s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122601_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:35:18][INFO][training:793] - 'deepspeed.initialize' took: 409.37578s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-134324_768_x4705c2s1b0n0.hostmgmt2705.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:51:19][INFO][training:795] - 'deepspeed.initialize' took: 393.09134s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-134324_768_x4705c2s1b0n0.hostmgmt2705.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:51:19][INFO][training:795] - 'deepspeed.initialize' took: 393.09134s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-141802_768_x4706c2s0b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:27:50][INFO][training:795] - 'deepspeed.initialize' took: 411.72035s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-165713_768_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:06:47][INFO][training:800] - 'deepspeed.initialize' took: 389.15768s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124517_768_x4315c4s1b0n0.hostmgmt2315.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:57:42][INFO][training:795] - 'deepspeed.initialize' took: 395.05079s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124517_768_x4315c4s1b0n0.hostmgmt2315.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:57:42][INFO][training:795] - 'deepspeed.initialize' took: 395.05079s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-165713_768_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:06:47][INFO][training:800] - 'deepspeed.initialize' took: 389.15768s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-130702_768_x4420c6s7b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:15:22][INFO][training:795] - 'deepspeed.initialize' took: 412.60004s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-130702_768_x4420c6s7b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:15:22][INFO][training:795] - 'deepspeed.initialize' took: 412.60004s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122601_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:35:18][INFO][training:793] - 'deepspeed.initialize took: 409.37578s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122601_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:35:18][INFO][training:793] - 'deepspeed.initialize took: 409.37578s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-173730_768_x4707c5s6b0n0.hostmgmt2707.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:45:33][INFO][training:800] - 'deepspeed.initialize' took: 400.74402s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124517_768_x4315c4s1b0n0.hostmgmt2315.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:57:42][INFO][training:795] - 'deepspeed.initialize' took: 395.05079s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-130702_768_x4420c6s7b0n0.hostmgmt2420.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:15:22][INFO][training:795] - 'deepspeed.initialize' took: 412.60004s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-132452_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:34:32][INFO][training:795] - 'deepspeed.initialize' took: 413.55765s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-173730_768_x4707c5s6b0n0.hostmgmt2707.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:45:33][INFO][training:800] - 'deepspeed.initialize' took: 400.74402s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-173730_768_x4707c5s6b0n0.hostmgmt2707.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:45:33][INFO][training:800] - 'deepspeed.initialize' took: 400.74402s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-132452_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:34:32][INFO][training:795] - 'deepspeed.initialize' took: 413.55765s
  ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb24576_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-132452_768_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:34:32][INFO][training:795] - 'deepspeed.initialize' took: 413.55765s
  ```

  </details>

- <details closed><summary><code>WORLD_SIZE = 1536</code>:</summary>

  ```bash
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-162028_1536_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:37:49][INFO][training:800] - 'deepspeed.initialize' took: 789.30077s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-162028_1536_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:37:49][INFO][training:800] - 'deepspeed.initialize' took: 789.30077s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-162028_1536_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:37:49][INFO][training:800] - 'deepspeed.initialize' took: 789.30077s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-145656_1536_x4119c5s7b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:14:35][INFO][training:795] - 'deepspeed.initialize' took: 788.86531s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-145656_1536_x4119c5s7b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:14:35][INFO][training:795] - 'deepspeed.initialize' took: 788.86531s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-145656_1536_x4119c5s7b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:14:35][INFO][training:795] - 'deepspeed.initialize' took: 788.86531s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122207_1536_x4309c6s4b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:39:56][INFO][training:793] - 'deepspeed.initialize' took: 792.71864s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122207_1536_x4309c6s4b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:39:56][INFO][training:793] - 'deepspeed.initialize' took: 792.71864s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-122207_1536_x4309c6s4b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 12:39:56][INFO][training:793] - 'deepspeed.initialize' took: 792.71864s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-125001_1536_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:06:47][INFO][training:795] - 'deepspeed.initialize' took: 836.98388s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-125001_1536_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:06:47][INFO][training:795] - 'deepspeed.initialize' took: 836.98388s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-175213_1536_x4702c1s4b0n0.hostmgmt2702.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 18:10:54][INFO][training:800] - 'deepspeed.initialize' took: 801.20500s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-175213_1536_x4702c1s4b0n0.hostmgmt2702.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 18:10:54][INFO][training:800] - 'deepspeed.initialize' took: 801.20500s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-184503_1536_x4702c1s4b0n0.hostmgmt2702.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 19:04:07][INFO][training:800] - 'deepspeed.initialize' took: 801.15950s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-131641_1536_x4315c4s1b0n0.hostmgmt2315.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:33:00][INFO][training:795] - 'deepspeed.initialize' took: 801.11322s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-213107_1536_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 21:46:29][INFO][training:800] - 'deepspeed.initialize' took: 820.95380s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-155216_1536_x4706c2s3b0n0.hostmgmt2706.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 16:07:23][INFO][training:795] - 'deepspeed.initialize' took: 787.04806s
  ws1536_ds_stage1_nl32_hs4096_mb4_seq4096_gb49152_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-141727_1536_x4102c7s0b0n0.hostmgmt2102.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:34:20][INFO][training:795] - 'deepspeed.initialize' took: 809.36787s
  ```

  </details>

- <details closed><summary><code>WORLD_SIZE = 3072</code>:</summary>

  ```bash
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-144534_3072_x4309c6s2b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:18:41][INFO][training:795] - 'deepspeed.initialize' took: 1639.62374s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-144534_3072_x4309c6s2b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:18:41][INFO][training:795] - 'deepspeed.initialize' took: 1639.62374s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-144534_3072_x4309c6s2b0n0.hostmgmt2309.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 15:18:41][INFO][training:795] - 'deepspeed.initialize' took: 1639.62374s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-213304_3072_x4704c0s6b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:03:15][INFO][training:800] - 'deepspeed.initialize' took: 1591.34487s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-213304_3072_x4704c0s6b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:03:15][INFO][training:800] - 'deepspeed.initialize' took: 1591.34487s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-213304_3072_x4704c0s6b0n0.hostmgmt2704.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 22:03:15][INFO][training:800] - 'deepspeed.initialize' took: 1591.34487s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-170636_3072_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:37:20][INFO][training:800] - 'deepspeed.initialize' took: 1632.12712s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-170636_3072_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:37:20][INFO][training:800] - 'deepspeed.initialize' took: 1632.12712s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-170636_3072_x4415c2s3b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 17:37:20][INFO][training:800] - 'deepspeed.initialize' took: 1632.12712s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124519_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:16:22][INFO][training:795] - 'deepspeed.initialize' took: 1674.44393s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124519_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:16:22][INFO][training:795] - 'deepspeed.initialize' took: 1674.44393s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-124519_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 13:16:22][INFO][training:795] - 'deepspeed.initialize' took: 1674.44393s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-140113_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:30:23][INFO][training:795] - 'deepspeed.initialize' took: 1618.10035s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-140113_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:30:23][INFO][training:795] - 'deepspeed.initialize' took: 1618.10035s
  ws3072_ds_stage1_nl32_hs4096_mb4_seq4096_gb98304_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240623-140113_3072_x4119c5s3b0n0.hostmgmt2119.cm.aurora.alcf.anl.gov/output.log:
      [2024-06-23 14:30:23][INFO][training:795] - 'deepspeed.initialize' took: 1618.10035s
  ```

  </details>

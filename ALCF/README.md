# Megatron-DeepSpeed @ ALCF

> [!IMPORTANT]
> This is a (backup) branch created on 2025-09-08, prior to merging the PR from
> [#88](https://github.com/argonne-lcf/Megatron-DeepSpeed/pull/88)

---

> [!IMPORTANT]
> [`train_aGPT_7B.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_aGPT_7B.sh) is the main entry point for launching
> distributed training on {Polaris, Aurora, Sunspot} @ ALCF.

## 🏃‍♂️ Running

1. Clone the [argonne-lcf / `Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed) repository:

    ```bash
    git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    cd Megatron-DeepSpeed
    ```

1. Set up your environment:

    ```bash
    export PBS_O_WORKDIR=$(pwd)
    source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
    ezpz_setup_env
    ```

    <details closed><summary>[Optional: Setup WandB]</summary>

    To enable [Weights & Biases](https://wandb.ai/) (WandB) logging,
    we need to install and login:

    ```bash
    python3 -m pip install wandb --upgrade
    wandb login
    ```

    > **NOTE**: WandB can be disabled by setting `export WANDB_DISABLED=1`

    See [`wandb`: Quickstart](https://docs.wandb.ai/quickstart) for
    additional information

   </details>


1. Install dependencies:

    1. 🍋 [saforem2 / `ezpz`](https://github.com/saforem2/ezpz):

       ```bash
       python3 -m pip install "git+https://github.com/saforem2/ezpz" --require-virtualenv
       ```

    1. [microsoft / `DeepSpeed`](https://github.com/microsoft/DeepSpeed):

       ```bash
       python3 -m pip install deepspeed --require-virtualenv
       ```

1. Launch training:

    ```bash
    # Before launching, `PBS_O_WORKDIR` should be set to Megatron-DeepSpeed's PATH
    # and venv inside Megatron-DeepSpeed/venv should be activated.
    PBS_O_WORKDIR=$(pwd) bash train_aGPT_7B.sh
    ```

    This will default to using the default AuroraGPT-7B architecture with the
    full [Dolma (v1.7)](https://huggingface.co/datasets/allenai/dolma) dataset.

    <details closed><summary>[Overridable Options]:</summary>

    This is a simple subset of the overridable options.

    The full list (as well as their default values) can be found in [ALCF / `helpers.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh)

    - `DTYPE`: Data type
    - `DATA_FILE_LIST`: Data file list
    - `FFN_HIDDEN_SIZE`: Feedforward Neural Network projection size
    - `GRAD_ACC_STEPS`: Gradient accumulation steps
    - `HEADS`: Number of attention heads
    - `HIDDEN`: Hidden size
    - `MICRO_BATCH`: Micro batch size
    - `NO_FLASH_ATTN`: No Flash Attention
    - `NLAYERS`: Number of layers
    - `NUM_KV_HEAD`: Number of key-value heads
    - `OPT`: Optimizer
        - `adam`
        - `adam8bit`
        - `adamw`
        - `adamwschedulefree`
        - `apex.adam`
        - `apex.sgd`
        - `ds.fusedlamb`
        - `ds.onebitlamb`
        - `galoreadamw`
        - `galoreadamw8bit`
        - `galoreadamw8bitperlayer`
        - `ipex.fusedlamb`
        - `ipex.lamb`
        - `shampoo`
        - `sgd`
        - `sgdschedulefree`
        - `sophiag`
    - `PP`: Pipeline parallelism degree
    - `SEQ`: Sequence length
    - `SP`: Sequence parallelism (Ulysses) degree
    - `TP`: Tensor parallelism degree
    - `TRAIN_TOKENS`: Number of training tokens
    - `TRAIN_ITERS`: Number of training iterations
    - `USE_ACTIVATION_CHECKPOINTING`: Use activation checkpointing
    - `WEIGHT_DECAY`: Weight decay
    - `ZERO_STAGE`: Zero stage
  
   </details>


### 🚀 Submit as a batch job

```bash
$ cd Megatron-DeepSpeed
$ qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home train_aGPT_7B.sh
```


## 📝 Data Preprocessing 

<details closed><summary>Data Pre-Processing:</summary>

AuroraGPT is trained on the Dolma dataset (initially v0), now in the process of moving to v6. For more details on the dataset, refer to https://huggingface.co/datasets/allenai/dolma. The dolma dataset downloaded is already preprocessing to remove the duplicates (dedup) and filtering the data (mixing). For more details refer to https://github.com/allenai/dolma/tree/main/docs and https://github.com/vksastry/dolma_alcf/blob/main/ALCF/Readme.md. 

The data preprocessing of Dolma dataset before training consists of tokenization of the data using a specific tokenizer (LlamaTokenizer is what we are currently using), Use the below script to tokenize the entire dataset. Example shown for Polaris. 

``` bash
cd /eagle/datasets/dolma/utils
./tokenization.sh
``` 

</details>

## ✅ TODOs

<details closed>
<summary>TODOs:</summary>

- [ ] Ensure / double check that optimizer settings from `ds_config.json` aren't being overwritten by some defaults in `megatron/arguments.py`
    - [ ] specifically, `momentum, beta{1, 2}, etc`

<details closed><summary><b>✅ <code>Completed</code></b></summary>

- Continue runs on Polaris @
    - [x] 48 Nodes
    - [x] 32 Nodes
    - [x] 16 Nodes
    - [x] 8 Nodes
    - [x] 4 Nodes

- [x] Then, try re-creating ( / fixing) conda with `cuda==12.1`
    - 😔, failed.

- ~~‼️  Unable to save checkpoints with `torch==2.1` + `cuda==11.8`~~:
    - Fixed in [a57a21f](https://github.com/argonne-lcf/Megatron-DeepSpeed/commit/a57a21f6b2a8abf847f5ef599e1b1edcb5a5e1b5)

    <details closed><summary><code>🐛 Bug</code></summary>

    - Training progresses OK:

        ```bash
        [2024-03-07 15:27:02,646] [INFO] [timer.py:260:stop] epoch=0/micro_step=199/global_step=199, RunningAvgSamplesPerSec=58.730622229657506, CurrSamplesPerSec=61.35304005128382, MemAllocated=6.01GB, MaxMemAllocated=19.52GB
        iteration      199/  317892 | consumed samples:       152832 | consumed tokens:    625999872 | elapsed time per iteration (ms): 14287.5 | learning rate: 2.407E-04 | global batch size:   768 | lm loss: 5.905366E+00 | loss scale: 8192.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 53.753 | tokens per gpu per second (tgs): 1146.733 | TFLOPs: 69.85 |
        [2024-03-07 15:27:15,063] [INFO] [logging.py:96:log_dist] [Rank 0] step=200, skipped=4, lr=[0.000240653265864008, 0.000240653265864008], mom=[(0.9, 0.999), (0.9, 0.999)]
        [2024-03-07 15:27:17,188] [INFO] [timer.py:260:stop] epoch=0/micro_step=200/global_step=200, RunningAvgSamplesPerSec=58.730745476291396, CurrSamplesPerSec=58.75503515561452, MemAllocated=6.01GB, MaxMemAllocated=19.52GB
        iteration      200/  317892 | consumed samples:       153600 | consumed tokens:    629145600 | elapsed time per iteration (ms): 14541.4 | learning rate: 2.407E-04 | global batch size:   768 | lm loss: 5.897035E+00 | loss scale: 8192.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 52.815 | tokens per gpu per second (tgs): 1126.713 | TFLOPs: 68.63 |
        saving checkpoint at iteration     200 to checkpoints/ds_stage2_nl32_hs4096_mb8_seq4096_gb768_pp1_tp2_fp16
        # ...
        ```

    - Then crashes with:

      ```python
      Traceback (most recent call last):
      Traceback (most recent call last):
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/pretrain_gpt_alcf.py", line 575, in <module>
          model = main()
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/pretrain_gpt_alcf.py", line 554, in main
          model = pretrain(
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 226, in pretrain
          iteration = train(forward_step_func,
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 1290, in train
          save_checkpoint_and_time(iteration, model, optimizer,
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/training.py", line 1151, in save_checkpoint_and_time
          save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/checkpointing.py", line 259, in save_checkpoint
          state_dict[UNIVERSAL_CHECKPOINT_INFO] = _universal_checkpoint_info(model)
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/checkpointing.py", line 783, in _universal_checkpoint_info
          info.update(model[0].universal_checkpoint_info())
        File "/lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/megatron/model/gpt_model.py", line 203, in universal_checkpoint_info
          info[TP_REPLICATED_PARAMETER_PATTERNS] = self._get_tp_replicated_param_patterns()
        File "/lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/2024-03-06/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1695, in __getattr__
          raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
      AttributeError: 'GPTModel' object has no attribute '_get_tp_replicated_param_patterns'
      ```

      🤔
</details>

</details>

</details>

</details>

</details>

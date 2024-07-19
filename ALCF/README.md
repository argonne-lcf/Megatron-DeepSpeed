# Megatron-DeepSpeed @ ALCF

> [!IMPORTANT]
> [`train_llama_alcf.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_llama_alcf.sh) is the main entry point for launching
> distributed training on {Polaris, Aurora, Sunspot} @ ALCF.

## 🏃‍♂️ Running

To launch on {`Polaris`, `Aurora`, `Sunspot`} @ [ALCF](https://alcf.anl.gov):

1. <details closed><summary>⏳ Request an interactive job with <code>qsub -I</code>:</summary>

    ```bash
    qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home -I
    ```

    - Or, alternatively, you can submit [`train_llama_alcf.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_llama_alcf.sh)
      directly as a batch script with

        ```bash
        cd Megatron-DeepSpeed
        qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00:filesystems=eagle:home train_llama_alcf.sh
        ```

</details>

2. <details closed><summary>⬇️ Clone repo + navigate into it:</summary>

    ```bash
    git clone "https://github.com/argonne-lcf/Megatron-DeepSpeed"
    cd Megatron-DeepSpeed
    ```

</details>

3. <details closed><summary>🐍 Setup Python:</summary>

    <br>

    > **NOTE**: The following commands should be ran from [`Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed), following the `cd` command from 2.

    1. Load `conda` module and activate base environment:

       ```bash
       export PBS_O_WORKDIR=$(pwd) && source ALCF/helpers.sh && setup_python
       ```

        - <details closed><summary><code>[output]</code>:</summary>

            <br>

            - <details closed><summary><code>[Polaris]</code>:</summary>

                ```bash
                # [05:47:13 PM][foremans@x3001c0s13b1n0][/eagle/a/f/p/ar/Megatron-DeepSpeed-D/Megatron-DeepSpeed]
                $ PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh && setup_python
                Using WORKING_DIR: /eagle/argonne_tpc/foremans/projects/argonne-lcf/Megatron-DeepSpeed-DistributedDataLoading/Megatron-DeepSpeed
                No conda_prefix or virtual_env found in environment...
                Setting up conda...
                Running on Polaris !!

                Lmod is automatically replacing "nvhpc/23.9" with "gcc-native/12.3".


                Lmod is automatically replacing "PrgEnv-nvhpc/8.5.0" with "PrgEnv-gnu/8.5.0".


                Due to MODULEPATH changes, the following have been reloaded:
                  1) cray-mpich/8.1.28

                Found conda at: /soft/applications/conda/2024-04-29/mconda3
                No VIRTUAL_ENV found in environment!
                    - Trying to setup from /soft/applications/conda/2024-04-29/mconda3
                    - Using VENV_DIR=/eagle/argonne_tpc/foremans/projects/argonne-lcf/Megatron-DeepSpeed-DistributedDataLoading/Megatron-DeepSpeed/venvs/2024-04-29
                    - Found existing venv, activating from /eagle/argonne_tpc/foremans/projects/argonne-lcf/Megatron-DeepSpeed-DistributedDataLoading/Megatron-DeepSpeed/venvs/2024-04-29
                [python] Using: /eagle/argonne_tpc/foremans/projects/argonne-lcf/Megatron-DeepSpeed-DistributedDataLoading/Megatron-DeepSpeed/venvs/2024-04-29/bin/python3
                ```

               </details>

           - <details closed><summary><code>[Aurora]</code>:</summary>

                ```bash
                # [10:04:02 PM][foremans@x4415c0s2b0n0][/gecko/A/fo/p/a/Megatron-DeepSpeed]
                $ PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh && setup_python
                Using WORKING_DIR: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed
                No conda_prefix or virtual_env found in environment...
                Setting up conda...

                The following have been reloaded with a version change:
                  1) intel_compute_runtime/release/821.36 => intel_compute_runtime/release/803.29     2) oneapi/eng-compiler/2024.04.15.002 => oneapi/release/2024.1

                Found conda at: /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1
                No VIRTUAL_ENV found in environment!
                    - Trying to setup from /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1
                    - Using VENV_DIR=/gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1
                    - Found existing venv, activating from /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1
                [python] Using: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/bin/python3
                ```

               </details>

           - <details closed><summary><code>[Sunspot]</code>:</summary>

                ```bash
                # [05:37:18 PM][foremans@x1921c0s0b0n0][/gila/A/fo/p/a/Megatron-DeepSpeed]
                $ PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh && setup_python
                Using WORKING_DIR: /gila/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed
                No conda_prefix or virtual_env found in environment...
                Setting up conda...
                Running on SunSpot !!

                Due to MODULEPATH changes, the following have been reloaded:
                  1) gcc/12.2.0             5) mpich-config/collective-tuning/1024
                  2) gmp/6.2.1-pcxzkau      6) mpich/icc-all-pmix-gpu/20231026
                  3) mpc/1.3.1-dfagrna      7) oneapi/eng-compiler/2024.04.15.002
                  4) mpfr/4.2.0-w7v7yjv

                The following have been reloaded with a version change:
                  1) intel_compute_runtime/release/821.36 => intel_compute_runtime/release/775.20
                  2) spack-pe-gcc/0.7.0-24.086.0 => spack-pe-gcc/0.6.1-23.275.2
                     UMD: agama-ci-devel-803.29 successfully loaded:
                     UMD: graphics-compute-runtime/agama-ci-devel-803.29

                The following have been reloaded with a version change:
                  1) oneapi/eng-compiler/2024.04.15.002 => oneapi/release/2024.04.15.001

                Found conda at: /soft/datascience/aurora_nre_models_frameworks-2024.1_preview_u1
                No VIRTUAL_ENV found in environment!
                    - Trying to setup from /soft/datascience/aurora_nre_models_frameworks-2024.1_preview_u1
                    - Using VENV_DIR=/gila/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1_preview_u1
                    - Found existing venv, activating from /gila/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1_preview_u1
                [python] Using: /lus/gila/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1_preview_u1/bin/python3
                ```

               </details>

    <!--
    3. Create virtual environment _on top of the base `conda`_[^venv]:

        ```bash
        export PBS_O_WORKDIR=$(pwd) && source ALCF/helpers.sh && setup_venv_from_conda
        ```
    -->

    2. 🍋 Install [`ezpz`](https://github.com/saforem2/ezpz):

        ```bash
        mkdir deps &&  git clone https://github.com/saforem2/ezpz deps/ezpz
        python3 -m pip install -e deps/ezpz --require-virtualenv
        ```

    [^venv]: Its generally a good practice to keep separate virtual Python environments different projects.  
        We provide a helper function, [`setup_venv_from_conda()`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/2f0154394bbdf3c64b4669f9d944645e2cdb8f2b/ALCF/helpers.sh#L440),
        that helps take care of this for you.  
        <br>
        This will: activate (or build, if necessary) a `venv` in your working dir,  
        _automatically_ matching the name of your active `conda` environment (e.g. `2024-04-29`, on Polaris_.

   3. Setup [`wandb`](https://docs.wandb.ai/quickstart)

      > **NOTE**: this can be disabled by setting `export WANDB_DISABLED=1`

</details>

<!--
Explicitly, it will (if inside a `conda` environment):

- look for a virtual environment in `"./venvs/${conda_tag}/"`
  (e.g. `./venvs/2024-04-29`) and:
    - if found:  
        - activate the existing virtual environment
    - else:
        - create a _new_ virtual environment in `"./venvs/${conda_tag}"`
            - activate it

Explicitly, at the command line:

```bash
PBS_O_WORKDIR=$(pwd) source ALCF/helpers.sh  # 1.
setup_conda_polaris    # 2.
setup_venv_from_conda  # 3.
```

will (1.) 
-->

4. <details closed><summary>🚀 Launch:</summary>

    In this case, train a ~ 2B Model (with 10 layers),
    for 1000 iterations using the data file list in:

    [`ALCF/data-lists/polaris/books.txt`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/data-lists/polaris/books.txt)

    with a micro-batch-size of 2 (`MICRO_BATCH=2`), with the `torch.optim.AdamW` optimizer (`OPT=adamw`).

    **Note** that _any_ of the options in the [`setParams`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh#L140)
    function from [`ALCF/helpers.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/7d203596dbf14e048e756c5ee6705de7dcb22283/ALCF/helpers.sh)
    can be overridden dynamically at runtime using this technique.

    ```bash
    # for systems other than Polaris, replace "polaris/books.txt" below with:
    # "{aurora,sunspot}/books.txt", 
    PBS_O_WORKDIR=$(pwd) DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt TRAIN_ITER=1000 NLAYERS=10 MICRO_BATCH=2 OPT=adamw bash train_llama_alcf.sh
    ```

    - **Note**: If no additional options specified, i.e.

        ```bash
        PBS_O_WORKDIR=$(pwd) bash train_llama_alcf.sh
        ```

      then this will fallback to using the default AuroraGPT-7B architecture
      with the full Dolma (v1.7) dataset.

    <details closed><summary><code>[output]</code>:</summary>

    <br>

    The outputs should look _something_ like this, though YMMV (things change quick):

    <details closed><summary><code>[Aurora]</code>:</summary>

    ```bash
    #[🌌][10:45:59 AM][foremans@x4711c1s2b0n0][…/Megatron-DeepSpeed][🌱 main][$!?]
    $ export PBS_O_WORKDIR=$(pwd) && source ALCF/helpers.sh && setup_python

    #[🌌][10:46:57 AM][foremans@x4711c1s2b0n0][…/Megatron-DeepSpeed][🌱 main][$!?][aurora_nre_models_frameworks-2024.1]
    (aurora_nre_models_frameworks-2024.1) $ PBS_O_WORKDIR=$(pwd) DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt bash train_llama_alcf.sh > train-log-$(tstamp).log 2>&1 &

    Using WORKING_DIR: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed
    Running on: aurora
    Using virtual_env: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1 on top of conda from: /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1
    [python] Using: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/bin/python3
    Ensuring all dependencies from /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/ALCF/requirements/requirements.txt installed...

    [notice] A new release of pip is available: 24.0 -> 24.1
    [notice] To update, run: pip install --upgrade pip
    ┌─────────────────────────────────────────────────────────────────────┐
    │ [savejobenv]:
    │     • Writing PBS vars to: /home/foremans/.pbsenv
    └─────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │ [HOSTS]:
    │     • [host:0] - x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov
    │     • [host:1] - x4711c1s3b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov
    └─────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │ [DIST INFO]:
    │     • HOSTFILE=/var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    │     • NHOSTS=2
    │     • NGPU_PER_HOST=12
    │     • NGPUS=24
    └─────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │ [LAUNCH]:
    │     • To launch across all available GPUs, use:
    │       'launch' ( = mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov )
    └─────────────────────────────────────────────────────────────────────┘
    2024-06-21 10:47:09,771 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    2024-06-21 10:47:09,772 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
    2024-06-21 10:47:09,772 - numexpr.utils - INFO - NumExpr defaulting to 8 threads.
    /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/lib/python3.9/site-packages/pandas/core/computation/expressions.py:21: UserWarning: Pandas requires version '2.8.4' or n>
      from pandas.core.computation.check import NUMEXPR_INSTALLED
    /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1/lib/python3.9/runpy.py:127: RuntimeWarning: 'ezpz.jobs' found in sys.modules after import of package 'ezpz', but prior to execution of 'ezpz.jobs'; this may result in u>
      warn(RuntimeWarning(msg))
    [2024-06-21 10:47:10][INFO][jobs:366] - Caught PBS_JOBID='684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov', pbsnf=PosixPath('/var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov') from env. Saving jobenv!
    [2024-06-21 10:47:10][WARNING][jobs:117] - /home/foremans/PBS-jobs/684084  already in /home/foremans/PBS-jobs.log,  not appending !!
    [2024-06-21 10:47:10][INFO][jobs:192] - Saving job env to /home/foremans/PBS-jobs/684084/jobenv.sh
    [2024-06-21 10:47:10][INFO][jobs:220] - Saving job env to /home/foremans/PBS-jobs/684084/jobenv.json
    [2024-06-21 10:47:10][INFO][jobs:233] - Saving job env to /home/foremans/PBS-jobs/684084/jobenv.yaml
    [2024-06-21 10:47:10][INFO][jobs:137] - Saving job env to .jobenv file in  /home/foremans/PBS-jobs/684084/.jobenv
    [2024-06-21 10:47:10][INFO][jobs:137] - Saving job env to .jobenv file in  /lus/gecko/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/.jobenv
    [2024-06-21 10:47:10][WARNING][jobs:154] - To use launch alias, be sure to:  source /lus/gecko/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/.jobenv
    [2024-06-21 10:47:10][INFO][jobs:277] - Writing PBS env vars to  /home/foremans/PBS-jobs/684084 / jobenv{.sh, .yaml, .json}
    [2024-06-21 10:47:10][WARNING][jobs:281] - Run: source ./.jobenv in your current shell to set job variables
    [2024-06-21 10:47:10][INFO][jobs:374] -
    [DIST_INFO]:
      • DEVICE=xpu
      • DEVICE_ID=xpu:0
      • DISTRIBUTED_BACKEND=ccl
      • GPUS_PER_NODE=12
      • HOSTS=['x4711c1s2b0n0', 'x4711c1s3b0n0']
      • HOSTFILE=/var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
      • HOSTNAME=x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov
      • LOCAL_RANK=0
      • MACHINE=Aurora
      • NUM_NODES=2
      • NGPUS=24
      • NODE_ID=0
      • RANK=0
      • SCHEDULER=PBS
      • WORLD_SIZE_TOTAL=24
      • WORLD_SIZE_IN_USE=1
    [2024-06-21 10:47:10][CRITICAL][jobs:245] - To launch across ALL GPUs in your job, use:
    LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    creating alias launch=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    Found ezpz!

    [notice] A new release of pip is available: 24.0 -> 24.1
    [notice] To update, run: pip install --upgrade pip
    Done with ezpz.
    Not using flash-attn!!
    LR_ARGS: --lr 0.0003 --lr-decay-style cosine --lr-warmup-fraction 0.05
    DS_CONFIG: /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/ds-configs/ds_stage1_mb4_gb768_pp1_bf16.json
    ZS: 1, MB: 4, GB: 768, PP: 1, DTYPE: bf16
     Please see logs at: logs/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240621-104713_24_x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov
    Checkpoints will be saved to: checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    Setting up tokenizer with Llama2
    Using data_file_list: ./ALCF/data-lists/aurora/books.txt
    Using tokenizer: Llama2. Setting up data with ./ALCF/data-lists/aurora/books.txt
    Calling:  setData() with ./ALCF/data-lists/aurora/books.txt
    --------------------
    Updated environment:
    DATA_FILE_LIST: ./ALCF/data-lists/aurora/books.txt
    NUM_DOCS: 3
     WEIGHT_SUM: 0.0072042092147565125
    DFL_STEM: books
    DATA_CACHE_PATH: .cache/books/index-cache
    DATA_FLAGS:  --data-file-list ./ALCF/data-lists/aurora/books.txt
    --------------------
    [setData] DATA_FLAGS:  --data-file-list ./ALCF/data-lists/aurora/books.txt
    [setData] TOKENIZER_FLAGS: --tokenizer-type Llama2Tokenizer --tokenizer-model /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/ALCF/tokenizer.model
    Requirement already satisfied: pybind11 in ./venvs/aurora_nre_models_frameworks-2024.1/lib/python3.9/site-packages (2.12.0)

    [notice] A new release of pip is available: 24.0 -> 24.1
    [notice] To update, run: pip install --upgrade pip
    make: Nothing to be done for 'default'.
    /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    - MPICH_DIR=/opt/aurora/24.086.0/CNDA/mpich/20231026/mpich-ofi-all-icc-default-pmix-gpu-drop20231026
    - Using /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/bin/python3
    - WORLD_SIZE:24
    - BACKEND: ccl
    - MODEL_TYPE: llama-seq4096-pp1-tp1-32layers-32heads-4096hidden
    - Using DATA_FILE_LIST: ./ALCF/data-lists/aurora/books.txt
    ++++++++++++++++++++++++++++++++++++++++++++++++++

    Currently Loaded Modules:
      1) mpich/icc-all-pmix-gpu/20231026       3) libfabric/1.15.2.0   5) cray-libpals/1.3.3            7) gmp/6.2.1-pcxzkau    9) mpc/1.3.1-dfagrna  11) intel_compute_runtime/release/803.29  13) frameworks/2024.1
      2) mpich-config/collective-tuning/1024   4) cray-pals/1.3.3      6) spack-pe-gcc/0.7.0-24.086.0   8) mpfr/4.2.0-w7v7yjv  10) gcc/12.2.0         12) oneapi/release/2024.1



    Saving environment to checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.env
    Not currently running. Continuing!
    Launching with: MPICH
     mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --genvall --cpu-bind depth -d 16 /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/bin/python3 -Wignore /lus/gecko/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/pretrain_gpt_alcf.py
    Using data_cache_path: checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache

            mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --genvall --cpu-bind depth -d 16 /gecko/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.1/bin/python3 -Wignore /lus/gecko/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/pretrain_gpt_alcf.py         --bf16                 --split 100,0,0         --log-interval 1         --no-bias-gelu-fusion         --no-bias-dropout-fusion         --no-masked-softmax-fusion         --no-gradient-accumulation-fusion        >

    [!! NOTE] View output at:
     logs/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/20240621-104713_24_x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov/output.log
    Connected to tcp://x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov:7919
    Launching application eafe3e80-ad2e-4cee-a3e4-d63af2a77c66
    [2024-06-21 10:47:31,610] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
    [2024-06-21 10:47:31,610] [INFO] [comm.py:637:init_distributed] cdb=None
    [2024-06-21 10:47:31,610] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=15, local_rank=3, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=23, local_rank=11, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=12, local_rank=0, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=13, local_rank=1, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=3, local_rank=3, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=14, local_rank=2, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=16, local_rank=4, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=17, local_rank=5, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=18, local_rank=6, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=19, local_rank=7, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=20, local_rank=8, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=21, local_rank=9, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=22, local_rank=10, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=6, local_rank=6, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=8, local_rank=8, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=10, local_rank=10, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend ccl
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=1, local_rank=1, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=2, local_rank=2, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=4, local_rank=4, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=5, local_rank=5, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=7, local_rank=7, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=9, local_rank=9, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:31,611] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=11, local_rank=11, world_size=24, master_addr=10.115.79.12, master_port=29500
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
    [2024-06-21 10:47:32][INFO][dist:291] - [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
    2024-06-21 10:47:32][INFO][dist:240] - DistInfo={
        "DEVICE": "xpu",
        "DEVICE_ID": "xpu:0",
        "DISTRIBUTED_BACKEND": "ccl",
        "GPUS_PER_NODE": 12,
        "HOSTFILE": "/var/spool/pbs/aux/684084.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "HOSTNAME": "x4711c1s2b0n0.hostmgmt2711.cm.aurora.alcf.anl.gov",
        "HOSTS": "['x4711c1s2b0n0', 'x4711c1s3b0n0']",
        "LOCAL_RANK": 0,
        "MACHINE": "Aurora",
        "NGPUS": 24,
        "NODE_ID": 0,
        "NUM_NODES": 2,
        "RANK": 0,
        "SCHEDULER": "PBS",
        "WORLD_SIZE_IN_USE": 24,
        "WORLD_SIZE_TOTAL": 24
    }

    # [...clipped...]

    [2024-06-21 10:48:48][INFO][utils:307] - > elapsed time for building blendable dataset indices: 1.19 (sec)
    [2024-06-21 10:48:48][INFO][utils:307] -  > saving index map files
    [2024-06-21 10:48:51][INFO][utils:307] -  > finished saving index map files in 3.0829622745513916 seconds
    [2024-06-21 10:48:51][INFO][utils:307] - > loading blendable dataset index: checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache/49e9529a32d0a98f1e40f4a82872b11c_index.npy
    [2024-06-21 10:48:52][INFO][utils:307] - > loading blendable dataset sample index: checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache/49e9529a32d0a98f1e40f4a82872b11c_sample_index.npy
    [2024-06-21 10:48:52][INFO][utils:307] - > finished loading in 0.30188989639282227 seconds
    [2024-06-21 10:48:52][INFO][utils:307] -  >> building dataset for /gecko/Aurora_deployment/AuroraGPT/datasets/dolma/data_v1.7_Llama2Tokenizer/books-0002_text_document
    [2024-06-21 10:48:52][INFO][utils:307] -  > building dataset index ...
    [2024-06-21 10:48:52][INFO][utils:307] -     reading sizes...
    [2024-06-21 10:48:52][INFO][utils:307] -     reading pointers...
    [2024-06-21 10:48:52][INFO][utils:307] -     reading document index...
    [2024-06-21 10:48:52][INFO][utils:307] -     creating numpy buffer of mmap...
    [2024-06-21 10:48:52][INFO][utils:307] - /gecko/Aurora_deployment/AuroraGPT/datasets/dolma/data_v1.7_Llama2Tokenizer/books-0002_text_document.bin
    [2024-06-21 10:48:52][INFO][utils:307] -     creating memory view of numpy buffer...
    [2024-06-21 10:48:52][INFO][utils:307] -  > finished creating indexed dataset in 0.003112 seconds
    [2024-06-21 10:48:52][INFO][utils:307] -     number of documents: 7386
    [2024-06-21 10:48:52][INFO][utils:307] -  > dataset split:
    [2024-06-21 10:48:52][INFO][utils:307] -     train:
    [2024-06-21 10:48:52][INFO][utils:307] -      document indices in [0, 7386) total of 7386 documents
    [2024-06-21 10:48:52][INFO][utils:307] -     validation:
    [2024-06-21 10:48:52][INFO][utils:307] -      document indices in [7386, 7386) total of 0 documents
    [2024-06-21 10:48:52][INFO][utils:307] -     test:
    [2024-06-21 10:48:52][INFO][utils:307] -      document indices in [7386, 7386) total of 0 documents
    [2024-06-21 10:48:52][INFO][utils:307] -  > loading doc-idx mapping from checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache/1fa7757ef8907da21e1e1326705e7f3f_doc_idx.npy
    [2024-06-21 10:48:52][INFO][utils:307] -  > loading sample-idx mapping from checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache/1fa7757ef8907da21e1e1326705e7f3f_sample_idx.npy
    [2024-06-21 10:48:52][INFO][utils:307] -  > loading shuffle-idx mapping from checkpoints/ws24_ds_stage1_nl32_hs4096_mb4_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr0.0003_lwf0.05/.cache/books/index-cache/1fa7757ef8907da21e1e1326705e7f3f_shuffle_idx.npy
    [2024-06-21 10:48:52][INFO][utils:307] -     loaded indexed file in 0.008 seconds
    [2024-06-21 10:48:52][INFO][utils:307] -     total number of samples: 34196233
    [2024-06-21 10:48:52][INFO][utils:307] -     total number of epochs: 175
    [2024-06-21 10:48:52][INFO][utils:307] - > size of blendable dataset: 245361763 samples
    [2024-06-21 10:48:52][INFO][utils:307] -  >>> Finished building BlendableDataset in 4.613574266433716 seconds
    [2024-06-21 10:48:52][INFO][pretrain_gpt_alcf:579] - > finished creating GPT datasets. Took: 45730179865763.24219s
    [2024-06-21 10:48:53][INFO][training:88] - [after dataloaders are built] datetime=2024-06-21 10:48:53
    [2024-06-21 10:48:53][INFO][training:307] - done with setup ...
    [2024-06-21 10:48:53][INFO][training:313] - training ...
    (min, max) time across ranks (ms):
        model-and-optimizer-setup ......................: (63763.34, 63857.25)
        train/valid/test-data-iterators-setup ..........: (12936.53, 13432.64)
    [2024-06-21 10:48:53][INFO][training:88] - [before the start of training step] datetime=2024-06-21 10:48:53
    [2024-06-21 10:48:53,396] [INFO] [checkpointing.py:541:forward] Activation Checkpointing Information
    [2024-06-21 10:48:53,396] [INFO] [checkpointing.py:542:forward] ----Partition Activations False, CPU CHECKPOINTING False
    [2024-06-21 10:48:53,396] [INFO] [checkpointing.py:543:forward] ----contiguous Memory Checkpointing False with 32 total layers
    [2024-06-21 10:48:53,396] [INFO] [checkpointing.py:545:forward] ----Synchronization False
    [2024-06-21 10:48:53,396] [INFO] [checkpointing.py:546:forward] ----Profiling time in checkpointing False
    [2024-06-21 10:50:42,167] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 1867.64 | optimizer_gradients: 19.65 | optimizer_step: 46.07
    [2024-06-21 10:50:42,167] [INFO] [logging.py:96:log_dist] [Rank 0] step=1, skipped=0, lr=[1.887433467970254e-08, 1.887433467970254e-08], mom=[(0.9, 0.999), (0.9, 0.999)]
    [2024-06-21 10:50:42,167] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 25341.72 | bwd_microstep: 77707.38 | bwd_inner_microstep: 75751.84 | bwd_allreduce_microstep: 1955.54 | step_microstep: 2218.38
    [2024-06-21 10:50:42,168] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 25341.72 | bwd: 77707.38 | bwd_inner: 75751.84 | bwd_allreduce: 1955.54 | step: 2218.38
    [2024-06-21 10:50:42][INFO][training:1609] -  iteration=       1/  317892 | consumed_samples=         768 | consumed_tokens=     3145728 | elapsed_time_per_iteration_ms=108893.2 | learning_rate=1.88743e-08 | global_batch_size=  768 | lm loss=11.133188 | loss_scale=1.0 | actual_seqlen= 4096 | number_of_skipped_iterations=  0 | number_of_nan_iterations=  0 | samples_per_second=7.053 | tokens_per_gpu_per_second_tgs=1203.674 | [LM]-TFLOPs=49.66 | [DS]-TFLOPs=73.32 |
    [2024-06-21 10:50:42][INFO][utils:190] - [Rank 0] (after 1 iterations) memory (MB) | allocated: 18243.64111328125 | max allocated: 50664.2548828125 | reserved: 54556.0 | max reserved: 54556.0
    (min, max) time across ranks (ms):
        forward-backward ...............................: (106622.81, 106624.28)
        optimizer ......................................: (2221.02, 2234.98)
    ```

    </details>

    <details closed><summary><code>[Sunspot]</code>:</summary>

    ```bash
    # [09:07:32 AM][foremans@x1921c0s0b0n0][~/q/llm.devkit/Megatron-DeepSpeed][🌱 main][$!?]
    $ PBS_O_WORKDIR=$(pwd) DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt bash train_llama_alcf.sh
    source-ing /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/ALCF/helpers.sh
    Sourcing /home/foremans/q4-drop_sunspot/llm.devkit/setenv.sh...
         UMD: agama-ci-devel-736.9 successfully loaded:
         UMD: graphics-compute-runtime/agama-ci-devel-736.9 
    Lmod has detected the following error: The following module(s) are unknown: "gcc/12.1.0"

    Please check the spelling or version number. Also try "module spider ..."
    It is also possible your cache file is out-of-date; it may help to try:
      $ module --ignore_cache load "gcc/12.1.0"

    Also make sure that all modulefiles written in TCL start with the string #%Module

    Note: the module "intel_compute_runtime/release/agama-devel-647" cannot be unloaded because it was not loaded.

    Running on SunSpot !!
    [python] Using: /home/foremans/miniconda3/envs/q4-drop/bin/python3
    Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env
    Found ezpz!
    /lus/gila/projects/Aurora_deployment/foremans/locations/sunspot/projects/saforem2/ezpz/src/ezpz/__init__.py
    Has ezpz installed. Nothing to do.
    Done with ezpz.
    ┌───────────────────────────────────────────────────────────────────
    │ Writing PBS vars to /home/foremans/.pbsenv
    │ HOSTFILE: /var/spool/pbs/aux/8988430.amn-0001
    │ NHOSTS: 2
    │ NGPU_PER_HOST: 12 GPUs per host
    │ NGPUS: 24 GPUs total
    └───────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Hosts]: 
    │     • [host:0] - x1921c0s0b0n0.hostmgmt2000.cm.americas.sgi.com
    │     • [host:1] - x1921c0s1b0n0.hostmgmt2000.cm.americas.sgi.com
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [DIST INFO]: 
    │     • Loading job env from: /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/8988430.amn-0001
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 12
    │     • NGPUS (NHOSTS x NGPU_PER_HOST): 24
    │     • WORLD_SIZE: 24
    │     • DIST_LAUNCH: mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/8988430.amn-0001
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Launch]:
    │     • Use: 'launch' (=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/8988430.amn-0001)
    │       to launch job
    └──────────────────────────────────────────────────────────────────
    DS_CONFIG: ds_stage2_mb4_gb96_pp1_bf16.json
    ZS: 2, CPU_OPTIMIZER: , MB: 4, GB: 96, PP: 1, DTYPE: bf16!!!Please see logs at logs/ds_stage2_nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16/0404090742_x1921c0s0b0n0
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    Calling:  setData() with ./convergence_debug_small.txt
    --------------------
    Updated environment:
    DATA_FILE_LIST: ./convergence_debug_small.txt
    NUM_DOCS: 15
     WEIGHT_SUM: 15.0
    DFL_STEM: convergence_debug_small
    DATA_CACHE_PATH: /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache
    --------------------
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    - MPICH_DIR=
    - Using /home/foremans/miniconda3/envs/q4-drop/bin/python3
    - WORLD_SIZE:24
    - NCCL: nccl
    - MODEL_TYPE: llama-seq4096-pp1-tp1-32layers-32heads-4096hidden
    - Using DATA_FILE_LIST: ./convergence_debug_small.txt
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    ! Using /home/foremans/miniconda3/envs/q4-drop/bin/deepspeed
    /home/foremans/miniconda3/envs/q4-drop/bin/ds_report:4: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
      __import__('pkg_resources').require('deepspeed==0.12.3+6ea44d02')
    /home/foremans/miniconda3/envs/q4-drop/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you dont plan on using image function
    ality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torch
    vision` from source?
      warn(
    [2024-04-04 09:07:45,585] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
    [2024-04-04 09:07:45,818] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
    --------------------------------------------------
    DeepSpeed C++/CUDA extension op report
    --------------------------------------------------
    NOTE: Ops not installed will be just-in-time (JIT) compiled at
          runtime if needed. Op compatibility means that your system
          meet the required dependencies to JIT install the op.
    --------------------------------------------------
    JIT compiled ops requires ninja
    ninja .................. [OKAY]
    --------------------------------------------------
    op name ................ installed .. compatible
    --------------------------------------------------
    async_io ............... [NO] ....... [OKAY]
    cpu_adagrad ............ [NO] ....... [OKAY]
    cpu_adam ............... [NO] ....... [OKAY]
    flash_attn ............. [NO] ....... [OKAY]
    fused_adam ............. [NO] ....... [OKAY]
    quantizer .............. [NO] ....... [OKAY]
    transformer ............ [NO] ....... [OKAY]
    transformer_inference .. [NO] ....... [OKAY]
    utils .................. [NO] ....... [OKAY]
    --------------------------------------------------
    DeepSpeed general environment info:
    torch install path ............... ['/home/foremans/miniconda3/envs/q4-drop/lib/python3.9/site-packages/torch']
    torch version .................... 2.1.0a0+cxx11.abi
    deepspeed install path ........... ['/lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/DeepSpeed/deepspeed']
    deepspeed info ................... 0.12.3+6ea44d02, 6ea44d02, HEAD
    deepspeed wheel compiled w. ...... torch 2.1 
    shared memory (/dev/shm) size .... 503.18 GB

        deepspeed --hostfile /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/hostfile_deepspeed --launcher MPICH /lus/gila/projects/Aurora_deployment/
    foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/pretrain_gpt_alcf.py     --bf16     --optimizer adamw     --split 100,0,0     --log-interval 1     --no-bias-gelu-fusion     --lr-decay
    -style cosine     --no-bias-dropout-fusion     --no-masked-softmax-fusion     --tokenizer-type Llama2Tokenizer     --no-gradient-accumulation-fusion     --accumulate-allreduce-grads-in-fp32 
        --use-checkpoint-opt_param-scheduler     --tensorboard-dir checkpoints/ds_stage2_nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16/tensorboard     --log-timers-to-tensorboard     --log-optimizer
    -states-to-tensorboard     --lr 0.0003     --save checkpoints/ds_stage2_nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16     --load checkpoints/ds_stage2_nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16  
       --seq-length 4096     --num-layers 32     --hidden-size 4096     --train-iters 317892     --eval-iters 10     --distributed-backend ccl     --num-attention-heads 32     --save-interval 20
    0     --eval-interval 50000     --max-position-embeddings 4096     --micro-batch-size 4     --data-file-list ./convergence_debug_small.txt     --tensor-model-parallel-size 1     --global-bat
    ch-size 96     --pipeline-model-parallel-size 1     --num-key-value-heads 8     --data-cache-path /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/
    .cache/convergence_debug_small/index-cache     --ffn-hidden-size 11008     --tokenizer-model /home/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/ALCF/tokenizer.model     --no-query-
    key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear      --deepspeed-activation-checkpointing  --z
    ero-stage=2  --deepspeed_config=ds_stage2_mb4_gb96_pp1_bf16.json  --no-pipeline-parallel  --deepspeed       --checkpoint-activations --checkpoint-num-layers 1           |& tee logs/ds_stage2
    _nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16/0404090742_x1921c0s0b0n0/output.log

    [!! NOTE] View output at:
    logs/ds_stage2_nl32_hs4096_mb4_seq4096_gb96_pp1_tp1_bf16/0404090742_x1921c0s0b0n0/output.log

    # ...

    /gila/Aurora_deployment/AuroraGPT/datasets/dolma/data_Llama2Tokenizer/common-crawl/cc_en_middle/cc_en_middle-0051_text_document.bin
        creating memory view of numpy buffer...
     > finished creating indexed dataset in 0.010017 seconds
        number of documents: 1498927
     > dataset split:
        train:
         document indices in [0, 1498927) total of 1498927 documents
        validation:
         document indices in [1498927, 1498927) total of 0 documents
        test:
         document indices in [1498927, 1498927) total of 0 documents
     > loading doc-idx mapping from /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/bf90c74a625ac2ee4de6e1d6f7f84fbb_doc_idx.npy
     > loading sample-idx mapping from /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/bf90c74a625ac2ee4de6e1d6f7f84fbb_sample_idx.npy
     > loading shuffle-idx mapping from /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/bf90c74a625ac2ee4de6e1d6f7f84fbb_shuffle_idx.npy
        loaded indexed file in 0.056 seconds
        total number of samples: 2318461
        total number of epochs: 8
    > loading blendable dataset index: /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/3a426af74008c22f9db24db811aad6b7_index.npy
    > loading blendable dataset sample index: /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/3a426af74008c22f9db24db811aad6b7_sample_index.npy
    /home/foremans/miniconda3/envs/q4-drop/lib/python3.9/site-packages/torch/utils/data/dataloader.py:557: UserWarning: This DataLoader will create 2 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.

    [after dataloaders are built] datetime: 2024-04-04 09:09:27
    done with setup ...
    (min, max) time across ranks (ms):
        model-and-optimizer-setup ......................: (64818.18, 64858.22)
        train/valid/test-data-iterators-setup ..........: (1968.10, 2288.56)
    training ...
    [before the start of training step] datetime: 2024-04-04 09:09:27
    [2024-04-04 09:09:27,718] [INFO] [checkpointing.py:540:forward] Activation Checkpointing Information
    [2024-04-04 09:09:27,719] [INFO] [checkpointing.py:541:forward] ----Partition Activations False, CPU CHECKPOINTING False
    [2024-04-04 09:09:27,719] [INFO] [checkpointing.py:542:forward] ----contiguous Memory Checkpointing False with 32 total layers
    [2024-04-04 09:09:27,719] [INFO] [checkpointing.py:544:forward] ----Synchronization False
    [2024-04-04 09:09:27,719] [INFO] [checkpointing.py:545:forward] ----Profiling time in checkpointing False
    [2024-04-04 09:09:33][INFO][utils:145] - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2024-04-04 09:09:33][INFO][utils:148] - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
    [2024-04-04 09:09:33][INFO][utils:160] - NumExpr defaulting to 8 threads.
    ^[c[2024-04-04 09:09:53,311] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 884.11 | optimizer_gradients: 6.43 | optimizer_step: 23.44
    [2024-04-04 09:09:53,312] [INFO] [logging.py:96:log_dist] [Rank 0] step=1, skipped=0, lr=[0.00029999999999267505, 0.00029999999999267505], mom=[(0.9, 0.999), (0.9, 0.999)]
    [2024-04-04 09:09:53,313] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 6567.68 | bwd_microstep: 17950.36 | bwd_inner_microstep: 17711.20 | bwd_allreduce_microstep: 239.11 | step_microstep: 1139.27
    [2024-04-04 09:09:53,313] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 6567.66 | bwd: 17950.35 | bwd_inner: 17711.19 | bwd_allreduce: 239.11 | step: 1139.29
    [Rank 0] (after 1 iterations) memory (MB) | allocated: 18244.640625 | max allocated: 41299.50146484375 | reserved: 46764.0 | max reserved: 46764.0
     iteration        1/  317892 | consumed samples:           96 | consumed tokens:       393216 | elapsed time per iteration (ms): 25849.1 | learning rate: 3.000E-04 | global batch size:    96 | lm loss: 1.117136E+01 | loss scale: 1.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 3.714 | tokens per gpu per second(tgs): 633.832 | TFLOPs: 38.61 |
    [2024-04-04 09:10:13,619] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 327.85 | optimizer_gradients: 6.26 | optimizer_step: 23.60
    [2024-04-04 09:10:13,619] [INFO] [logging.py:96:log_dist] [Rank 0] step=2, skipped=0, lr=[0.00029999999997070033, 0.00029999999997070033], mom=[(0.9, 0.999), (0.9, 0.999)]
    [2024-04-04 09:10:13,620] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 4022.74 | bwd_microstep: 15738.67 | bwd_inner_microstep: 15556.80 | bwd_allreduce_microstep: 181.82 | step_microstep: 371.01
    [2024-04-04 09:10:13,620] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 4022.73 | bwd: 15738.66 | bwd_inner: 15556.62 | bwd_allreduce: 181.81 | step: 371.02
     iteration        2/  317892 | consumed samples:          192 | consumed tokens:       786432 | elapsed time per iteration (ms): 20298.3 | learning rate: 3.000E-04 | global batch size:    96 | lm loss: 2.537718E+01 | loss scale: 1.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.729 | tokens per gpu per second(tgs): 807.159 | TFLOPs: 49.17 |
    ```

    </details>

    <details closed><summary><code>[Polaris]</code>:</summary>

    ```bash
    # [09:31:35 AM][foremans@x3112c0s13b0n0][~/pol/p/a/Megatron-DeepSpeed][🌱 main][$!?]
    $ PBS_O_WORKDIR=$(pwd) DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt OPT=adamw bash train_llama_alcf.sh
    source-ing /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/ALCF/helpers.sh
    Running on Polaris !!

    [python] Using: /eagle/datascience/foremans/miniconda3/envs/cu118-pt221/bin/python3
    Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env
    Found ezpz!
    /lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/ezpz/src/ezpz/__init__.py
    Has ezpz installed. Nothing to do.
    Done with ezpz.
    ┌───────────────────────────────────────────────────────────────────
    │ Writing PBS vars to /home/foremans/.pbsenv
    │ HOSTFILE: /var/spool/pbs/aux/1822297.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    │ NHOSTS: 2
    │ NGPU_PER_HOST: 4 GPUs per host
    │ NGPUS: 8 GPUs total
    └───────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Hosts]: 
    │     • [host:0] - x3112c0s13b0n0.hsn.cm.polaris.alcf.anl.gov
    │     • [host:1] - x3112c0s13b1n0.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [DIST INFO]: 
    │     • Loading job env from: /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/1822297.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 4
    │     • NGPUS (NHOSTS x NGPU_PER_HOST): 8
    │     • WORLD_SIZE: 8
    │     • DIST_LAUNCH: mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1822297.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Launch]:
    │     • Use: 'launch' (=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1822297.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov)
    │       to launch job
    └──────────────────────────────────────────────────────────────────
    DS_CONFIG: ds_stage2_mb8_gb32_pp1_bf16.json
    ZS: 2, CPU_OPTIMIZER: , MB: 8, GB: 32, PP: 1, DTYPE: bf16!!!Please see logs at logs/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_pp1_tp2_bf16/0404093534_x3112c0s13b0n0
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    !! Caught USE_ACTIVATION_CHECKPOINTING=1 !!
    Calling:  setData() with "./convergence_debug_small.txt"
    --------------------
    Updated environment:
    DATA_FILE_LIST: ./convergence_debug_small.txt
    NUM_DOCS: 15
     WEIGHT_SUM: 15.0
    DFL_STEM: convergence_debug_small
    DATA_CACHE_PATH: /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache
    --------------------
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    - MPICH_DIR=/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1
    - Using /eagle/datascience/foremans/miniconda3/envs/cu118-pt221/bin/python3
    - WORLD_SIZE:8
    - NCCL: nccl
    - MODEL_TYPE: llama-seq4096-pp1-tp2-32layers-32heads-4096hidden
    - Using DATA_FILE_LIST: ./convergence_debug_small.txt
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    ! Using /eagle/datascience/foremans/miniconda3/envs/cu118-pt221/bin/deepspeed
    [2024-04-04 09:35:35,959] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda [auto detect]
    --------------------------------------------------
    DeepSpeed C++/CUDA extension op report
    --------------------------------------------------
    NOTE: Ops not installed will be just-in-time (JIT) compiled at
          runtime if needed. Op compatibility means that your system
          meet the required dependencies to JIT install the op.
    --------------------------------------------------
    JIT compiled ops requires ninja
    ninja .................. [OKAY]
    --------------------------------------------------
    op name ................ installed .. compatible
    --------------------------------------------------
    async_io ............... [NO] ....... [OKAY]
    fused_adam ............. [NO] ....... [OKAY]
    cpu_adam ............... [NO] ....... [OKAY]
    cpu_adagrad ............ [NO] ....... [OKAY]
    cpu_lion ............... [NO] ....... [OKAY]
     [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
    evoformer_attn ......... [NO] ....... [NO]
    fused_lamb ............. [NO] ....... [OKAY]
    fused_lion ............. [NO] ....... [OKAY]
    inference_core_ops ..... [NO] ....... [OKAY]
    cutlass_ops ............ [NO] ....... [OKAY]
    transformer_inference .. [NO] ....... [OKAY]
    quantizer .............. [NO] ....... [OKAY]
    ragged_device_ops ...... [NO] ....... [OKAY]
    ragged_ops ............. [NO] ....... [OKAY]
    random_ltd ............. [NO] ....... [OKAY]
     [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.2
     [WARNING]  using untested triton version (2.2.0), only 1.0.0 is known to be compatible
    sparse_attn ............ [NO] ....... [NO]
    spatial_inference ...... [NO] ....... [OKAY]
    transformer ............ [NO] ....... [OKAY]
    stochastic_transformer . [NO] ....... [OKAY]
    --------------------------------------------------
    DeepSpeed general environment info:
    torch install path ............... ['/eagle/datascience/foremans/miniconda3/envs/cu118-pt221/lib/python3.12/site-packages/torch']
    torch version .................... 2.2.1
    deepspeed install path ........... ['/eagle/datascience/foremans/miniconda3/envs/cu118-pt221/lib/python3.12/site-packages/deepspeed']
    deepspeed info ................... 0.14.0, unknown, unknown
    torch cuda version ............... 11.8
    torch hip version ................ None
    nvcc version ..................... 11.8
    deepspeed wheel compiled w. ...... torch 2.2, cuda 11.8
    shared memory (/dev/shm) size .... 251.61 GB

        deepspeed --hostfile /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/hostfile_deepspeed --launcher MPICH /lus/eagle/projects/datascienc
    e/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/pretrain_gpt_alcf.py     --bf16     --optimizer adamw     --split 100,0,0     --log-interval 1     --no-bias-gelu-fusion 
        --lr-decay-style cosine     --no-bias-dropout-fusion     --no-masked-softmax-fusion     --tokenizer-type Llama2Tokenizer     --no-gradient-accumulation-fusion     --accumulate-allreduce-
    grads-in-fp32     --use-checkpoint-opt_param-scheduler     --tensorboard-dir checkpoints/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_pp1_tp2_bf16/tensorboard     --log-timers-to-tensorboard     -
    -log-optimizer-states-to-tensorboard     --lr 0.0003     --save checkpoints/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_pp1_tp2_bf16     --load checkpoints/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_
    pp1_tp2_bf16     --seq-length 4096     --num-layers 32     --hidden-size 4096     --train-iters 317892     --eval-iters 10     --distributed-backend nccl     --num-attention-heads 32     --s
    ave-interval 200     --eval-interval 50000     --max-position-embeddings 4096     --micro-batch-size 8     --data-file-list ./convergence_debug_small.txt     --tensor-model-parallel-size 2  
       --global-batch-size 32     --pipeline-model-parallel-size 1     --num-key-value-heads 8     --data-cache-path /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-l
    cf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache     --ffn-hidden-size 11008     --tokenizer-model /home/foremans/polaris/projects/argonne-lcf/Megatron-DeepSpeed/ALCF/tokeniz
    er.model     --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --use-flash-attn-v2   
       --deepspeed-activation-checkpointing  --zero-stage=2  --deepspeed_config=ds_stage2_mb8_gb32_pp1_bf16.json  --no-pipeline-parallel  --deepspeed       --checkpoint-activations --checkpoint-
    num-layers 1           |& tee logs/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_pp1_tp2_bf16/0404093534_x3112c0s13b0n0/output.log

    [!! NOTE] View output at:
    logs/ds_stage2_nl32_hs4096_mb8_seq4096_gb32_pp1_tp2_bf16/0404093534_x3112c0s13b0n0/output.log

    # ...

    /eagle/datasets/dolma/data_Llama2Tokenizer/common-crawl/cc_en_middle/cc_en_middle-0051_text_document.bin
        creating memory view of numpy buffer...
     > finished creating indexed dataset in 0.001280 seconds
        number of documents: 1498927
     > dataset split:
        train:
         document indices in [0, 1498927) total of 1498927 documents
        validation:
         document indices in [1498927, 1498927) total of 0 documents
        test:
         document indices in [1498927, 1498927) total of 0 documents
     > loading doc-idx mapping from /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/9217d94f3290abc2fddf9e87bff236d6_doc_idx.npy
     > loading sample-idx mapping from /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/9217d94f3290abc2fddf9e87bff236d6_sample_idx.npy
     > loading shuffle-idx mapping from /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/9217d94f3290abc2fddf9e87bff236d6_shuffle_idx.npy
        loaded indexed file in 0.004 seconds
        total number of samples: 869423
        total number of epochs: 3
    > loading blendable dataset index: /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/a815d51f6752c6f486d94194ce95fb87_index.npy
    > loading blendable dataset sample index: /lus/eagle/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/.cache/convergence_debug_small/index-cache/a815d51f6752c6f486d94194ce95fb87_sample_index.npy
    > size of blendable dataset: 10223415 samples
    > finished creating GPT datasets ...
    [after dataloaders are built] datetime: 2024-04-04 09:36:07
    done with setup ...
    (min, max) time across ranks (ms):
        model-and-optimizer-setup ......................: (4794.78, 4795.23)
        train/valid/test-data-iterators-setup ..........: (589.69, 721.20)
    training ...
    [before the start of training step] datetime: 2024-04-04 09:36:07
    [2024-04-04 09:36:07,407] [INFO] [checkpointing.py:539:forward] Activation Checkpointing Information
    [2024-04-04 09:36:07,407] [INFO] [checkpointing.py:540:forward] ----Partition Activations False, CPU CHECKPOINTING False
    [2024-04-04 09:36:07,407] [INFO] [checkpointing.py:541:forward] ----contiguous Memory Checkpointing False with 32 total layers
    [2024-04-04 09:36:07,407] [INFO] [checkpointing.py:543:forward] ----Synchronization False
    [2024-04-04 09:36:07,407] [INFO] [checkpointing.py:544:forward] ----Profiling time in checkpointing False
    [2024-04-04 09:36:28,429] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 1626.54 | optimizer_gradients: 19.29 | optimizer_step: 419.48
    [2024-04-04 09:36:28,430] [INFO] [logging.py:96:log_dist] [Rank 0] step=1, skipped=0, lr=[0.00029999999999267505, 0.00029999999999267505], mom=[(0.9, 0.999), (0.9, 0.999)]
    [2024-04-04 09:36:28,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 11336.34 | bwd_microstep: 7134.73 | bwd_inner_microstep: 7090.02 | bwd_allreduce_microstep: 44.65 | step_microstep: 2564.02
    [2024-04-04 09:36:28,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 11336.33 | bwd: 7134.75 | bwd_inner: 7090.01 | bwd_allreduce: 44.66 | step: 2564.02
     iteration        1/  317892 | consumed samples:           32 | consumed tokens:       131072 | elapsed time per iteration (ms): 21133.8 | learning rate: 3.000E-04 | global batch size:    32 | lm loss: 1.119983E+01 | loss scale: 1.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.514 | tokens per gpu per second(tgs): 775.250 | TFLOPs: 47.23 |
    [Rank 1] (after 1 iterations) memory (MB) | allocated: 14165.525390625 | max allocated: 22332.37255859375 | reserved: 24642.0 | max reserved: 35824.0
    [Rank 0] (after 1 iterations) memory (MB) | allocated: 14165.525390625 | max allocated: 22332.37255859375 | reserved: 24642.0 | max reserved: 32994.0
    [2024-04-04 09:36:38,623] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 1605.55 | optimizer_gradients: 11.56 | optimizer_step: 50.92
    [2024-04-04 09:36:38,623] [INFO] [logging.py:96:log_dist] [Rank 0] step=2, skipped=0, lr=[0.00029999999997070033, 0.00029999999997070033], mom=[(0.9, 0.999), (0.9, 0.999)]
    [2024-04-04 09:36:38,623] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 1395.17 | bwd_microstep: 6832.48 | bwd_inner_microstep: 6789.73 | bwd_allreduce_microstep: 42.70 | step_microstep: 1867.64
    [2024-04-04 09:36:38,623] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 1395.15 | bwd: 6832.49 | bwd_inner: 6789.73 | bwd_allreduce: 42.71 | step: 1867.65
     iteration        2/  317892 | consumed samples:           64 | consumed tokens:       262144 | elapsed time per iteration (ms): 10154.3 | learning rate: 3.000E-04 | global batch size:    32 | lm loss: 1.766422E+01 | loss scale: 1.0 | actual seqlen:  4096 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 3.151 | tokens per gpu per second(tgs): 1613.503 | TFLOPs: 98.29 |

    # ...
    ```

    </details>

    </details>

</details>

<!--

[^example]: |
    In this case, train a ~ 2B Model (with 10 layers),
    for 1000 iterations using the data file list in:

    [`ALCF/data-lists/polaris/books.txt`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/data-lists/polaris/books.txt)

    with a micro-batch-size of 2, with the `torch.optim.AdamW` optimizer. Note that _any_ of the options in the

    [`setParams`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/helpers.sh#L140)

    function from

    [`ALCF/helpers.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/7d203596dbf14e048e756c5ee6705de7dcb22283/ALCF/helpers.sh)

    can be overridden dynamically at runtime using this technique.
-->

<!--
export PBS_O_WORKDIR="$(pwd)" && DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt bash train_llama_alcf.sh
export PBS_O_WORKDIR="$(pwd)" && DATA_FILE_LIST=./ALCF/data-lists/polaris/books.txt bash train_llama_alcf.sh
-->



<!--

## 📦 Install

<details closed><summary>Install Instructions</summary>

1. Clone [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

    ```bash
    $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    $ cd Megatron-DeepSpeed
    ```

     > [!NOTE]  
     > In the `conda create` command below,
     > you can replace `--name "${DAY}"` with
     > `--prefix /path/to/your/conda/envs`, if you prefer:

2. Create `conda` env:

    ```bash
    $ module load conda/2023-10-04
    $ export MPICC="cc -shared -taret-accel=nvidia80"
    $ export DAY=$(date "+%Y-%m-%d")
    $ export PYTHONUSERBASE="${HOME}/.local/polaris/conda/${DAY}"
    $ conda create --solver libmamba -c pytorch -c nvidia --name "${DAY}" "python==3.12"
    ```

3. Install dependencies:

    ```bash
    $ conda activate "${DAY}"  # e.g. 2024-03-07
    $ conda install -c pytorch -c nvidia --solver libmamba mpi4py ninja transformers xformers triton pytorch torchvision torchaudio pytorch-cuda=11.8
    $ conda install --solver libmamba mpi4py -c conda-forge -c pytorch -c nvidia
    $ python3 -m pip install --upgrade pip pybind11 toolong appdirs wandb sentencepiece ipython setuptools wheel ninja
    $ python3 -m pip install --upgrade deepspeed wandb
    ```

    - [`ezpz`](https://github.com/saforem2/ezpz):

        <details closed><summary><code>install</code>:</summary>

        ```bash
        $ git clone https://github.com/saforem2/ezpz
        $ python3 -m pip install -e "ezpz[dev]"
        ```

        </details>

     - [**OPTIONAL**] [`NVIDIA/apex`](https://github.com/NVIDIA/apex):

        <details closed><summary><code>install</code>:</summary>

        ```bash
        $ git clone https://github.com/NVIDIA/apex
        $ cd apex
        # NOTE: need GCC < 11 for APEX ¯\_(ツ)_/¯ ??
        $ module swap gcc gcc/10.3.0
        $ python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
        ```

        </details>

</details>

<!--
### Install

1. Clone [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

    ```bash
    $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    $ cd Megatron-DeepSpeed
    ```

2. Create `conda` env:

    ```bash
    $ module load conda/2023-10-04
    $ export MPICC="cc -shared -taret-accel=nvidia80"
    $ export DAY=$(date "+%Y-%m-%d")
    $ export PYTHONUSERBASE="${HOME}/.local/polaris/conda/${DAY}"
    $ conda create --solver libmamba -c pytorch -c nvidia --name "${DAY}" "python==3.10"
    ```

    > [!NOTE]
    > In the `conda create` command above,
    > you can replace `--name "${DAY}"` with
    > `--prefix /path/to/your/conda/envs`, if you prefer:

3. Install dependencies:

    ```bash
    $ conda activate "${DAY}"  # e.g. 2024-03-07
    $ conda install -c pytorch -c nvidia --solver libmamba mpi4py ninja transformers xformers triton pytorch torchvision torchaudio pytorch-cuda=11.8
    $ conda install --solver libmamba mpi4py -c conda-forge -c pytorch -c nvidia
    $ python3 -m pip install --upgrade pip pybind11 toolong appdirs wandb sentencepiece ipython setuptools wheel ninja
    $ python3 -m pip install --upgrade deepspeed wandb
    ```

    - [`NVIDIA/apex`](https://github.com/NVIDIA/apex):

        ```bash
        $ git clone https://github.com/NVIDIA/apex
        $ cd apex
        # NOTE: need GCC < 11 for APEX ¯\_(ツ)_/¯ ??
        $ module swap gcc gcc/10.3.0
        $ python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
        ```

    - [`ezpz`](https://github.com/saforem2/ezpz):

        ```bash
        $ git clone https://github.com/saforem2/ezpz
        $ python3 -m pip install -e "ezpz[dev]"
        ```
-->

<!--
### Running

- The (shell) script used to launch pre-training is:
    - [`train_llama_alcf.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_llama_alcf.sh)

- This shell script will set the appropriate environment variables, load the correct conda
modules and launch
[`pretrain_gpt_alcf.py`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/pretrain_gpt_alcf.py) using `mpiexec`

- Explicitly, to launch:

    ```bash
    # 1. Launch interactive job
    $ qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home -I
    # 2. Load conda environment
    $ module load conda/2023-10-04 ; conda activate /eagle/datascience/foremans/miniconda3/envs/cu118-pt221 ; unset PYTHONUSERBASE
    # 3. Navigate into `Megatron-DeepSpeed` directory
    $ cd Megatron-DeepSpeed
    # 4. Launch:
    $ export PBS_O_WORKDIR=$(pwd)
    $ bash train_llama_alcf_polaris.sh
    ```
    <details closed><summary><b>[Output]</b></summary>

    ```bash
    source-ing /lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/ALCF/helpers_alcf.sh

    CommandNotFoundError: Your shell has not been properly configured to use 'conda deactivate'.
    To initialize your shell, run

        $ conda init <SHELL_NAME>

    Currently supported shells are:
      - bash
      - fish
      - tcsh
      - xonsh
      - zsh
      - powershell

    See 'conda init --help' for more information and options.

    IMPORTANT: You may need to close and restart your shell after running 'conda init'.


    Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env
    Found ezpz!
    /lus/eagle/projects/datascience/foremans/tmp/Megatron-DeepSpeed/ezpz/src/ezpz/__init__.py
    Has ezpz installed. Nothing to do.
    ┌──────────────────────────────────────────────────────────────────
    │ [Hosts]:
    │     • [host:0] - x3005c0s37b0n0.hsn.cm.polaris.alcf.anl.gov
    │     • [host:1] - x3005c0s37b1n0.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [DIST INFO]:
    │     • Loading job env from: /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 4
    │     • NGPUS (NHOSTS x NGPU_PER_HOST): 8
    │     • WORLD_SIZE: 8
    │     • DIST_LAUNCH: mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Launch]:
    │     • Use: 'launch' (=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1777928.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov)
    │       to launch job
    └──────────────────────────────────────────────────────────────────
    # [...]
    ```
    </details>

-->


### 🚀 Submit as a batch job

```bash
$ cd Megatron-DeepSpeed
$ qsub -A <your-project> -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home train_llama_alcf.sh
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

# 🐛 Debugging Megatron-DeepSpeed[^sanity]

If you're running into issues with Megatron-DeepSpeed, here are some things to
try that I've found useful.

> [!NOTE]
> This guide assumes you're **running directly on a compute node**,
> and should work on _any_ of the ALCF systems[^alcf].
> (though it _should_ work anywhere, let me know if you run into issues!)

## 🔄 Turning it Off and On Again

The first thing to try is to simply re-run your script, after resetting your
environment (see below):

```bash
"${SHELL}" --login ; now=$(date +'%Y-%m-%d-%H%M%S'); debug_dir="${HOME}/debugging/${now}" && mkdir -p "${debug_dir}" && cd "${debug_dir}"&& git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed && source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env && python3 -m pip install --require-virtualenv "git+https://github.com/saforem2/ezpz" "numpy<2" deepspeed tensorboard && ezpz-test && DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
```

## 🤔 Why Does this Happen?

One of the most common issues encountered by users is a mangled environment.

This happens often when a user is:

- Loading system modules (`module load ...`)
- Trying to use `python`. In particular:
  - Activating `conda` environments (`conda activate ...`)
  - Using virtual environments, e.g.:
    (`uv venv`, `python -m venv`, `source venv/bin/activate`, ..., etc.)
  - Working on multiple projects with _different_ virtual environments

Ultimately, this is usually due to some combination of the above causing
conflicts in one (or more) of:

```bash
"${PATH}", "${LD_LIBRARY_PATH}", "${CUDA_HOME}",
"${VIRTUAL_ENV}", "${VENV_DIR}", "${PYTHONPATH}", "${CONDA_PREFIX}",
...
```

being misconfigured[^bad_env].

<!--
Some of the most common reasons this may happen are:

- Loading system modules (`module load ...`) that overwrite or silently change
  things in your active environment
  - These will often change your `PATH`, `LD_LIBRARY_PATH`, and other
    environment variables, taking precedence over your `conda` environment or
    other installed software
- Activating `conda` environments (`conda activate ...`)



- `module load` commands
- `conda activate` commands
- `pip install --user ...` commands
- `source /path/to/some/setup/script.sh` commands
-->

[^bad_env]: Among _many_ possible others.

## 🧪 Things to Try

1. **Reset your environment**: If you're in an interactive session, you can get
   a clean environment by re-logging into the node:

   ```bash
   ssh $(hostname)
   ```

   - <details closed><summary>Example:</summary>

      ```bash
      #[/f/d/f/p/s/ezpz][🌱 main][📦📝🤷✓]
      #[07/14/25 @ 07:35:04][x4301c6s1b0n0]
      ; export TEST_VAR=1

      #[/f/d/f/p/s/ezpz][🌱 main][📦📝🤷✓]
      #[07/14/25 @ 07:35:09][x4301c6s1b0n0]
      ; ssh $(hostname)
      Last login: Mon Jul 14 12:30:56 2025 from aurora-uan-0010.hostmgmt1000.cm.aurora.alcf.anl.gov

      #[~][C v7.5.0-gcc]
      #[07/14/25 @ 07:35:33][x4301c6s1b0n0]
      ; echo "${TEST_VAR}"


      #[~][C v7.5.0-gcc]
      #[07/14/25 @ 07:35:35][x4301c6s1b0n0]
      ;
      Connection to x4301c6s1b0n0 closed.
      took: 0h:00m:19s

      #[/f/d/f/p/s/ezpz][🌱 main][📦📝🤷✓] [⏱️ 19s]
      #[07/14/25 @ 07:35:37][x4301c6s1b0n0]
      ; echo "${TEST_VAR}"
      1
      ```

   </details>

1. **Start from scratch**:
   - Create a _new_, _isolated_ directory for debugging
  
     ```bash
     now=$(date +'%Y-%m-%d %H:%M:%S')
     debug_dir="debugging/${now}"
     mkdir -p "${debug_dir}" && cd "${debug_dir}"
     echo "Debugging in $(pwd)"
  
   - Create a new clone of the repository

     ```bash
     git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed
     ```
     
   - Create a new virtual environment

     ```bash
     source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env
     ```
     
   - Re-install dependencies

     ```bash
     python3 -m pip install -e "git+https://github.com/saforem2/ezpz"
     ```

   - Run simple test to verify python can launch distributed processes:
  
     ```bash
     ezpz-test
     ```
     
   - Try re-running
  
     ```bash
     DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
     ```

[^alcf]: Yes, _any_ of the ALCF systems! e.g.: Aurora, Polaris, ThetaGPU, Sunspot, Sophia, Sirius, ...

[^sanity]: While trying to maintain your sanity 😂

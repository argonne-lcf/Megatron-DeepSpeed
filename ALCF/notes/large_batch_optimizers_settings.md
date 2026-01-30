# Megatron-DeepSpeed, optimizers, hyperparameters
`**Important** For large batch training with infinite schedulers, It is crucial to tune the learning rate as these schedulers benefit from larger learning rate values. See below for the `lr_finder` routine implemented in MDS to do so.`

Single command to test and run Megatron-DeepSpeed:

```bash
now=$(date +'%Y-%m-%d-%H%M%S') && debug_dir="${now}" && mkdir -p "${debug_dir}"&& cd "${debug_dir}"&& git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed && source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env && python3 -m pip install --require-virtualenv "git+https://github.com/saforem2/ezpz" "numpy<2" deepspeed tensorboard && ezpz-test && DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
```
## Optimizers
The default optimizer is `adamw`. Go to this [list of optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485) for a complete list of supported optimizers (note that dshampoo might throw checkpointing errors, we are working on fixing this). For example, to run with `muon`, you can do:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
Here
```bash
DATA_FILE_LIST=path/to/your/tokenized/data
TRAIN_TOKENS= number of training tokens
GRAD_ACC_STEPS=number of grad accumulation steps
LR=learning rate
LR_WARMUP_FRACTION=warmup fraction
OPT=optimizer
```
Your global batch size will be: `num_gpus*micro_batch_size*GRAD_ACC_STEPS`, micro batch size is 1 by default, you can change it by adding `MICRO_BATCH=new_micro_batch_size` to your options. To have the corresponding number if tokens per step, you need to multiply the global batch size by the sequence length (set with `SEQ_LEN`, default is 4096)

### Adding custom optimizers
To add a custom optimizer, you have to modify the following files:
- `megatron/optimizer/__init__.py`: [muon example](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/optimizer/__init__.py#L434), note that you either heve to import the optimizer from a pre-installed package or add it in the `megatron/optimizer/` folder.
- `megatron/arguments.py`: [optimizer arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1070), to add the optimizer arguments
- `megatron/arguments.py`: [list of valid optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485), to add the new optimizer to the list of valid optimizers

### Schedulers
Note that the default scheduler is `cosine`. We also support `infinite cosine, infinite inverse square root, constant, constant with cooldown, inverse square root, linear` schedulers. For example to change the scheduler to `constant`, you can do so with the `LR_DECAY_STYLE` option:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
To add cooldown, you need to add the `--lr_constant_plus_cooldown` flag and set the cooldown fraction with `--lr_constant_plus_cooldown_frac`. The default cooldown fraction is 0.05
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --lr_constant_plus_cooldown_frac 0.01
```
#### Adding custom schedulers
To add a custom scheduler, you have to modify the following files:
- `megatron/optimizer_param_scheduler.py`: [schedulers](megatron/optimizer_param_scheduler.py), to add the new scheduler
- `megatron/arguments.py`: [list of LR arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1671), to add the new scheduler arguments.
- You might have to change [the function](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/training.py#L559) to incorporate your custom scheduler options.

## Hyperparameter tuning
#### Init variance
Weight initialization is key to training LLMs,and to avoid spikes in losses. Here, we initialize the weights following this [paper](https://arxiv.org/pdf/2312.16903). The default variance value at initialization is 0.02. To add custom variances, one can use `--init-method-std, `--adjust-word-embedding-init`, and `--word-embedding-init-std`. For our runs, we do
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632
```
where `d=hidden size`. In general, the initialization should be 
```bash
--init-method-std sqrt{ 2 / (5 * d) }  --adjust-word-embedding-init --word-embedding-init-std sqrt{ 2 / 5 }
```
### Learning rate
For the learning rate, we implemented the learning rate finder routine [here](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) and [here](https://arxiv.org/pdf/1506.01186). This is activated with the `--lr-finder` and run for `TRAIN_ITERS` steps. For example, for a 1000 steps:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_ITERS=1000 GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632 --lr-finder
```
This approach allows to find the largest LR one can train with without the model divergence. Training with large LR is crucial for infinite schedulers as well for large batch training. We increase the LR following a power law at each step and monitor the LR curve in particular the decaying phase and the blow up phase. To tune the LR, you identify the point where the LR start increasing and divide that LR value by 10. You can also pick the LR corresponding to the steepest descent phase. The learning rates are stored in the output folder (the one set in **SAVE=**).
![lr_finder](./assets/lb_optimizers/lr_finder_example.png)
After running the LR-finder routine, you can modify the code below to plot it and find suitable LR candidates:
```bash

# ---------- Helper: find ALL local minima vs log10(LR) ----------
def find_all_minima_lrs(learning_rates, losses, smooth_frac=0.03, min_log_sep=0.0):
    """
    Return ALL learning rates where the loss switches from decreasing->increasing
    (local minima of loss with respect to log10(LR)).

    Parameters
    ----------
    learning_rates : array-like
    losses         : array-like
    smooth_frac    : fraction of length for moving-average smoothing (reduce noise).
                     Try 0.02–0.05 if needed.
    min_log_sep    : minimum spacing in log10(LR) between reported minima.
                     Set 0.0 to return all raw minima (no de-dupe).

    Returns
    -------
    list of LRs (floats), sorted ascending (left→right).
    """
    lr = np.asarray(learning_rates, float)
    y  = np.asarray(losses, float)

    # ensure LR is increasing
    order = np.argsort(lr)
    lr, y = lr[order], y[order]

    # light smoothing to avoid jitter creating fake minima
    w = max(5, int(len(y) * smooth_frac) | 1)  # odd window
    if w >= len(y):
        w = (len(y) - 1) | 1
    y_s = np.convolve(y, np.ones(w) / w, mode="same")

    # first derivative w.r.t. log10(LR) to match your log-x axis
    x  = np.log10(lr)
    dy = np.gradient(y_s, x)

    # adaptive epsilon so tiny flat noise doesn't cause false flips
    eps = 1e-12 + 0.02 * np.median(np.abs(dy))

    # minima where slope crosses 0 from negative -> positive
    neg = dy[:-1] < -eps
    pos = dy[1:]  >  eps
    idx = np.where(neg & pos)[0]

    # sub-sample minimum position by interpolating where dy == 0
    xmins = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        y0, y1 = dy[i], dy[i+1]
        denom = (y1 - y0)
        xz = x0 if denom == 0 else x0 - y0 * (x1 - x0) / denom
        if np.isfinite(xz):
            xmins.append(xz)

    # optional de-dup: enforce separation in log space (default 0.0 → keep all)
    xmins = sorted(xmins)
    if min_log_sep > 0.0 and len(xmins) > 1:
        kept = [xmins[0]]
        for xz in xmins[1:]:
            if all(abs(xz - p) >= min_log_sep for p in kept):
                kept.append(xz)
        xmins = kept

    return [10**px for px in xmins]

# ---------- Safe savefig wrapper (uses your savefig if present) ----------
def savefig_safe(fig, name):
    try:
        if 'savefig' in globals() and callable(globals()['savefig']):
            globals()['savefig'](fig, name, None)
        else:
            fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
    except Exception:
        fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")

# Small utility to optionally place a dot at a given x if it lies inside data range
def scatter_if_in_range(ax, x_target, x_all, y_all, color, size=45, zorder=6):
    x_min, x_max = np.min(x_all), np.max(x_all)
    if (x_target >= x_min) and (x_target <= x_max):
        y_val = np.interp(np.log10(x_target), np.log10(x_all), y_all)
        ax.scatter(x_target, y_val, color=color, s=size, zorder=zorder)

# =====================================================
# ===============  LAMB (optional)  ===================
# =====================================================
try:
    fig, ax = plt.subplots(figsize=(18, 12))
    i = 0
    for opt in optims_lamb:
        fname = opt + '/lr_finder_agpt_olmo/lr_finder/lr_finder_data.npz'
        data = np.load(fname)
        learning_rates = data['learning_rates'][:430]
        losses = data['losses'][:430]

        ax.plot(learning_rates, losses, linewidth=4, label=opt, color=f'C{i+1}')

        # absolute minimum (diamond)
        min_idx = np.argmin(losses)
        min_lr = learning_rates[min_idx]
        ax.scatter(min_lr, losses[min_idx], s=100, color=f'C{i+1}', marker='D', zorder=5)

        # ALL minima where loss turns up
        minima_lrs = find_all_minima_lrs(learning_rates, losses,
                                         smooth_frac=0.03, min_log_sep=0.0)  # set >0 to de-dupe
        suggested_lrs = [lr / 10.0 for lr in minima_lrs]

        print(f"{opt} minima LRs (all):     ", [f"{lr:.3e}" for lr in minima_lrs])
        print(f"{opt} suggested LRs (min/10):", [f"{lr:.3e}" for lr in suggested_lrs])

        # draw minima (solid dashed) + suggested (dotted)
        for lrj in minima_lrs:
            #ax.axvline(x=lrj, color=f'C{i+1}', linestyle='--', alpha=0.6, linewidth=1.8)
            scatter_if_in_range(ax, lrj, learning_rates, losses, color=f'C{i+1}')
        for lrj in suggested_lrs:
            ax.axvline(x=lrj, color=f'C{i+1}', linestyle=':', alpha=0.5, linewidth=1.6,label=f"{opt} suggested LR: {lrj:.2e}")
            scatter_if_in_range(ax, lrj, learning_rates, losses, color=f'C{i+1}', size=35, zorder=6)
        ax.axvline(x=0.0002, color=f'C{i+1}', linestyle='-', alpha=0.5, linewidth=1.6,label=f"{opt} current LR: 0.0002")
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder Comparison (LAMB)', fontsize=32)
        ax.legend(fontsize=14)
        i += 1

    savefig_safe(fig, 'figs_agpt/lr_finder_agpt_olmo_lamb')
    plt.show()
except NameError:
    # optims_lamb not defined; skip
    pass
    


```
### Maximal Update Parametrization/Complete Parametrization
We have MuP and CompleteP incorporated in AuroraGPT in the `lb-optimizers` branch.

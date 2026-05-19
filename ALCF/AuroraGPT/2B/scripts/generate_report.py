#!/usr/bin/env python3
"""Regenerate the AuroraGPT-2B Pre-Training W&B report locally.

Pulls run histories from the `aurora_gpt/AuroraGPT` W&B project using the same
filter as the public report
(https://api.wandb.ai/links/aurora_gpt/zyabwz9i),
groups runs the same way the report does (machine / NHOSTS / torch_version /
GBS / optimizer / data_file_list), and emits one PNG per panel into
ALCF/AuroraGPT/2B/assets/.

Run from the repo root:

    .venv-report/bin/python ALCF/AuroraGPT/2B/scripts/generate_report.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ambivalent  # noqa: F401  -- registers style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

plt.style.use(ambivalent.STYLES["ambivalent"])

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "ALCF" / "AuroraGPT" / "2B" / "assets"
CACHE_DIR = REPO_ROOT / "ALCF" / "AuroraGPT" / "2B" / ".cache"
ENTITY = "aurora_gpt"
PROJECT = "AuroraGPT"

# Same 22 filter clauses as the report.
REPORT_FILTERS: dict[str, Any] = {
    "$and": [
        {"createdAt": {"$gte": "2025-08-20T05:00:00.000Z"}},
        {"username": "foremans"},
        {"config.args.value.optimizer": "sophiag"},
        {"config.args.value.rope_theta": {"$in": [50000]}},
        {"config.args.value.zero_stage": 0},
        {"config.args.value.checkpoint_activations": False},
        {"config.args.value.micro_batch_size": 1},
        {"config.args.value.world_size": {"$in": [3072, 6240]}},
        {"config.args.value.num_layers": {"$in": [4, 6, 8, 10, 12, 14, 16, 18, 20]}},
        {"config.args.value.tokenizer_model":
            {"$in": ["google/gemma-7b", "google/gemma-7B"]}},
        {"config.args.value.deepspeed_config_dict.gradient_accumulation_steps":
            {"$gte": 2}},
        {"displayName": {"$nin": [
            "treasured-thunder-4350",
            "grateful-dust-4351",
            "glowing-disco-4352",
        ]}},
    ]
}

# (filename, title, x_metric, y_metric, x_label, y_label, x_log, y_log)
PANELS = [
    ("loss_lm_loss_vs_tokens",       "LM loss vs. consumed tokens",
     "training/consumed_tokens", "loss/lm loss",
     "Consumed tokens", "LM loss", False, True),
    ("loss_lm_loss_vs_tokens_linear", "LM loss vs. consumed tokens (linear)",
     "training/consumed_tokens", "loss/lm loss",
     "Consumed tokens", "LM loss", False, False),
    ("grad_norm_vs_tokens",          "Gradient norm vs. consumed tokens",
     "training/consumed_tokens", "loss/grad_norm",
     "Consumed tokens", "Gradient norm", False, True),
    ("val_lm_loss_vs_iter",          "Validation LM loss vs. iteration",
     "val/iteration", "val/lm loss",
     "Validation iteration", "Val LM loss", False, False),
    ("iter_time_vs_runtime",         "Iteration time vs. wall time",
     "_runtime", "training/iteration_time",
     "Wall time (min)", "Iteration time (s)", False, False),
    ("lr_vs_iter_linear",            "Learning rate vs. iteration",
     "learning-rate/iteration", "optimizer/learning_rate",
     "Iteration", "Learning rate", False, False),
    ("lr_vs_iter_loglog",            "Learning rate vs. iteration (log-log)",
     "learning-rate/iteration", "optimizer/learning_rate",
     "Iteration", "Learning rate", True, True),
    ("params_in_billions_vs_time",   "Approx. parameters (B) vs. wall time",
     "_timestamp", "throughput/approx_params_in_billions",
     "Date", "Parameters (B)", False, False),
    ("tokens_per_sec_vs_iter",       "Tokens/sec vs. iteration",
     "throughput/iteration", "throughput/tokens_per_sec",
     "Iteration", "Tokens / sec", False, False),
    ("tflops_lm_vs_runtime",         "TFLOPs (LM) vs. wall time",
     "_runtime", "throughput/tflops-lm",
     "Wall time (min)", "TFLOPs (LM)", False, False),
    ("consumed_tokens_vs_time",      "Consumed tokens vs. wall time",
     "_timestamp", "training/consumed_tokens",
     "Date", "Consumed tokens", False, False),
    ("tokens_per_gpu_per_sec_vs_iter", "Tokens/GPU/sec vs. iteration",
     "throughput/iteration", "throughput/tokens_per_gpu_per_sec",
     "Iteration", "Tokens / GPU / sec", False, False),
]


def get_v(d: dict, key: str, default: Any = None) -> Any:
    """Read a nested key from a run.config dict.

    Accepts the report's ``foo.value.bar`` paths *and* the flat keys the
    Python API returns (``foo.bar``). The ``.value`` segments are inserted by
    W&B's GraphQL representation but stripped by the Python SDK.
    """
    cur: Any = d
    for part in key.split("."):
        if part == "value":
            continue
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part, default)
    return cur


def fetch_runs(api: wandb.Api) -> list[wandb.apis.public.Run]:
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=REPORT_FILTERS, per_page=100))
    print(f"matched {len(runs)} runs")
    return runs


def label_for(run: wandb.apis.public.Run) -> str:
    cfg = run.config
    torch_ver = get_v(cfg, "torch_version") or "?"
    gbs = get_v(cfg, "args.value.global_batch_size") or "?"
    opt = get_v(cfg, "args.value.optimizer") or "?"
    theta = get_v(cfg, "args.value.rope_theta") or "?"
    return f"[torch:{torch_ver}][gbs:{gbs}][opt:{opt}][θ:{theta}]"


def group_key(run: wandb.apis.public.Run) -> tuple:
    cfg = run.config
    return (
        get_v(cfg, "machine"),
        get_v(cfg, "env.value.NHOSTS"),
        get_v(cfg, "torch_version"),
        get_v(cfg, "args.value.global_batch_size"),
        get_v(cfg, "args.value.optimizer"),
        get_v(cfg, "args.value.data_file_list"),
    )


def cache_path(run_id: str, pair_key: str) -> Path:
    return CACHE_DIR / run_id / f"{pair_key}.parquet"


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def load_pair(
    run: wandb.apis.public.Run, x_metric: str, y_metric: str,
) -> pd.DataFrame:
    """Load a single (x, y) metric pair for a run with on-disk caching.

    Each (x, y) pair is fetched separately because W&B's ``run.history`` uses
    an inner join over the requested keys and drops every step where any
    requested metric is missing — and val/* metrics, training/* metrics, and
    throughput/* metrics are logged at different cadences in these runs.
    """
    pair_key = f"{_safe_name(x_metric)}__VS__{_safe_name(y_metric)}"
    fp = cache_path(run.id, pair_key)
    if fp.exists():
        return pd.read_parquet(fp)
    keys = [x_metric] if x_metric in {"_runtime", "_timestamp"} else [x_metric, y_metric]
    if x_metric in {"_runtime", "_timestamp"}:
        keys = [y_metric]
        df = run.history(keys=keys, pandas=True, samples=10_000, x_axis=x_metric)
    else:
        df = run.history(keys=[x_metric, y_metric], pandas=True, samples=10_000)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fp)
    return df


def plot_panel(
    runs_by_group: dict[tuple, list[wandb.apis.public.Run]],
    fname: str,
    title: str,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    x_log: bool,
    y_log: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drew_any = False
    cmap = plt.get_cmap("tab10")
    for i, (gkey, runs_in_group) in enumerate(runs_by_group.items()):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for run in runs_in_group:
            try:
                df = load_pair(run, x_metric, y_metric)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {run.id} {x_metric}/{y_metric}: {e}")
                continue
            if df.empty or x_metric not in df.columns or y_metric not in df.columns:
                continue
            sub = df[[x_metric, y_metric]].dropna()
            if sub.empty:
                continue
            xs.append(sub[x_metric].to_numpy())
            ys.append(sub[y_metric].to_numpy())
        if not xs:
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        order = np.argsort(x)
        x, y = x[order], y[order]
        if x_metric == "_runtime":
            x = x / 60.0
        if x_metric == "_timestamp":
            x = pd.to_datetime(x, unit="s")
        # The first 5 grouping axes are constant for these runs, so the
        # data-file-list is the only one worth showing in the legend.
        dfl = gkey[5] or "(unknown data)"
        dfl_short = Path(dfl).name if dfl else "?"
        ax.plot(x, y, label=dfl_short, color=cmap(i % 10), linewidth=1.2)
        drew_any = True
    if not drew_any:
        plt.close(fig)
        print(f"  [skip] {fname}: no data points")
        return
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if len(runs_by_group) <= 8:
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    out = OUT_DIR / f"{fname}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    api = wandb.Api()
    runs = fetch_runs(api)

    runs_by_group: dict[tuple, list[wandb.apis.public.Run]] = defaultdict(list)
    for r in runs:
        runs_by_group[group_key(r)].append(r)
    print(f"grouped into {len(runs_by_group)} group(s)")

    for panel in PANELS:
        print(f"panel {panel[0]}...")
        plot_panel(runs_by_group, *panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())

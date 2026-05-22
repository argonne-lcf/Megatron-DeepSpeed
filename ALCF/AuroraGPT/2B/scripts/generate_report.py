#!/usr/bin/env python3
"""Regenerate the AuroraGPT-2B Pre-Training W&B report locally.

Pulls run histories from the `aurora_gpt/AuroraGPT` W&B project using the same
filter as the public report
(https://api.wandb.ai/links/aurora_gpt/zyabwz9i),
groups runs the same way the report does (machine / NHOSTS / torch_version /
GBS / optimizer / data_file_list), and emits one SVG per panel into
ALCF/AuroraGPT/2B/assets/. The exact (x, y) arrays plotted in each panel are
also dumped to ALCF/AuroraGPT/2B/data/<panel>.parquet so anyone can rebuild
the figures (or re-plot in a different tool) without hitting the W&B API.

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

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "ALCF" / "AuroraGPT" / "2B" / "assets"
DATA_DIR = REPO_ROOT / "ALCF" / "AuroraGPT" / "2B" / "data"
CACHE_DIR = REPO_ROOT / "ALCF" / "AuroraGPT" / "2B" / ".cache"

# Pick the first Iosevka family that is locally installed (if any).
_IOSEVKA = None
for _font in ("Iosevka Custom", "Iosevka", "Iosevka IBM", "Iosevka Nerd Font"):
    if any(f.name == _font for f in plt.matplotlib.font_manager.fontManager.ttflist):
        _IOSEVKA = _font
        break

THEMES: dict[str, dict] = {
    # Background stays transparent for both — only the text/axis color
    # changes so the figures sit on either a light or a dark page.
    "light": {"fg": "#262626"},
    "dark":  {"fg": "#f8f8f8"},
}


def apply_theme(theme: str) -> None:
    plt.style.use(ambivalent.STYLES["ambivalent"])
    if _IOSEVKA:
        plt.rcParams["font.family"] = _IOSEVKA
    fg = THEMES[theme]["fg"]
    plt.rcParams.update({
        # ~1.75x the ambivalent defaults
        "font.size": 20,
        "font.weight": "medium",
        "axes.titlesize": 22,
        "axes.titleweight": "semibold",
        "axes.labelsize": 20,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
        "figure.titlesize": 24,
        "figure.titleweight": "semibold",
        # Text / axis colors only — leave the backgrounds alone so the
        # SVG ships with no fill behind the plot area.
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.edgecolor": fg,
        "axes.titlecolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
    })
ENTITY = "aurora_gpt"
PROJECT = "AuroraGPT"

# Per-panel zoom insets keyed by filename:
#   xlim, ylim, bounds = (left, bottom, width, height) in axes fraction
INSETS: dict[str, dict] = {}

# Per-panel fixed y-axis limits keyed by filename.
YLIMS: dict[str, tuple[float, float]] = {
    # Steady-state iter time is ~3.4 s, p99 is ~26 s. 50 s shows the bulk
    # plus moderate spikes without being dominated by the few outliers
    # at 2500+ s (visible on the un-zoomed panel).
    "iter_time_vs_runtime_zoom": (0, 50),
}

# Per-panel stride for `[::N]` downsampling, keyed by filename.
DOWNSAMPLE: dict[str, int] = {
    "tokens_per_sec_vs_iter": 5,
    "tokens_per_gpu_per_sec_vs_iter": 5,
    "tflops_lm_vs_runtime": 5,
}

# Panels where we want a scatter (markers, no connecting line) instead of
# the default line plot. Iteration time is dominated by occasional spikes
# from many overlapping runs; drawing them as lines fills the canvas with
# spurious vertical segments connecting unrelated samples.
SCATTER_PANELS: set[str] = {
    "iter_time_vs_runtime",
    "iter_time_vs_runtime_zoom",
}

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
        {"displayName": {"$nin": [
            "happy-surf-4362",
            "laced-puddle-4445",
            "sage-universe-4443",
            "dandy-sea-4440",
            "twilight-dream-4439",
            "vibrant-glitter-4358",
            "dulcet-forest-4361",
            "copper-pyramid-4359",
            "lemon-aardvark-4360",
            "sleek-tree-4390",
            "autumn-sea-4394",
        ]}},
        {"tags": {"$nin": ["cooldown"]}},
        {"config.args.value.data_file_list":
            {"$nin": [
                "ALCF/data-lists/aurora/books.txt",
                "ALCF/data-lists/aurora/nvidia-math1-code2.txt",
            ]}},
    ]
}

# (filename, title, x_metric, y_metric, x_label, y_label, x_log, y_log)
PANELS = [
    ("loss_lm_loss_vs_tokens",       "Training loss vs. consumed tokens",
     "training/consumed_tokens", "loss/lm loss",
     "Consumed tokens", "Training loss", False, True),
    ("loss_lm_loss_vs_tokens_linear", "Training loss vs. consumed tokens (linear)",
     "training/consumed_tokens", "loss/lm loss",
     "Consumed tokens", "Training loss", False, False),
    ("grad_norm_vs_tokens",          "Gradient norm vs. consumed tokens",
     "training/consumed_tokens", "loss/grad_norm",
     "Consumed tokens", "Gradient norm", False, True),
    ("val_lm_loss_vs_iter",          "Validation loss vs. iteration",
     "val/iteration", "val/lm loss",
     "Validation iteration", "Validation loss", False, False),
    ("iter_time_vs_runtime",         "Iteration time vs. wall time",
     "_runtime", "training/iteration_time",
     "Wall time (min)", "Iteration time (s)", False, False),
    ("iter_time_vs_runtime_zoom",    "Iteration time vs. wall time (zoom)",
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
    ("tflops_lm_vs_runtime",         "TFLOPs vs. wall time",
     "_runtime", "throughput/tflops-lm",
     "Wall time (min)", "TFLOPs", False, False),
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
    theme: str,
) -> None:
    out_dir = ASSETS_DIR / theme
    fig, ax = plt.subplots(figsize=(12, 6.5))
    drew_any = False
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    panel_rows: list[dict] = []
    stride = DOWNSAMPLE.get(fname, 1)
    use_scatter = fname in SCATTER_PANELS
    for gkey, runs_in_group in runs_by_group.items():
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
        if stride > 1:
            x, y = x[::stride], y[::stride]
        if x_metric == "_runtime":
            x = x / 60.0
        if x_metric == "_timestamp":
            x = pd.to_datetime(x, unit="s")
        dfl = gkey[5] or "(unknown data)"
        dfl_short = Path(dfl).name if dfl else "?"
        if use_scatter:
            # Big enough to read individually. No edge color because the
            # background is transparent — we use a contrasting (foreground)
            # edge with reduced alpha so overlapping points stay
            # distinguishable on either light or dark pages.
            sc = ax.scatter(
                x, y,
                s=28, alpha=0.55, label=dfl_short,
                linewidths=0.6, edgecolors=THEMES[theme]["fg"],
            )
            color = sc.get_facecolor()[0]
            from matplotlib.colors import to_hex
            color = to_hex(color[:3])
        else:
            line, = ax.plot(x, y, label=dfl_short, linewidth=1.2)
            color = line.get_color()
        curves.append((x, y, color))
        drew_any = True
        # Record what we just plotted (post-sort, post-stride, post-unit-conversion)
        # so the parquet on disk matches the figure 1-to-1.
        if hasattr(x, "to_numpy"):
            x_save = x.to_numpy()
        else:
            x_save = np.asarray(x)
        panel_rows.append(pd.DataFrame({
            "group_data_file": dfl_short,
            "color": color,
            "x": x_save,
            "y": y,
        }))
    if not drew_any:
        plt.close(fig)
        print(f"  [skip] {fname}: no data points")
        return
    # Title moves into the legend area when the legend lives above the
    # axes, so drop the redundant title.
    if len(runs_by_group) > 8:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if fname in YLIMS:
        ax.set_ylim(*YLIMS[fname])
    if len(runs_by_group) <= 8:
        # Park the legend above the axes so it never sits on top of the data.
        leg = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(runs_by_group), 4),
            frameon=False,
            handlelength=2.0,
        )
        # Make scatter markers full-opacity in the legend even though the
        # scatter dots themselves are translucent.
        if use_scatter:
            for handle in leg.legend_handles:
                handle.set_alpha(1.0)
                handle.set_sizes([40])

    inset_cfg = INSETS.get(fname)
    if inset_cfg is not None:
        axins = ax.inset_axes(inset_cfg["bounds"])
        for x, y, color in curves:
            if use_scatter:
                axins.scatter(
                    x, y,
                    s=24, alpha=0.55, color=color,
                    linewidths=0.5, edgecolors=THEMES[theme]["fg"],
                )
            else:
                axins.plot(x, y, color=color, linewidth=1.0)
        axins.set_xlim(*inset_cfg["xlim"])
        axins.set_ylim(*inset_cfg["ylim"])
        axins.tick_params(labelsize=11)
        ax.indicate_inset_zoom(axins, edgecolor="0.4", alpha=0.6, linewidth=0.8)

    fig.tight_layout()
    out = out_dir / f"{fname}.svg"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")

    # The data is identical across themes, so only dump it once.
    if theme == "light" and panel_rows:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        panel_df = pd.concat(panel_rows, ignore_index=True)
        data_out = DATA_DIR / f"{fname}.parquet"
        panel_df.to_parquet(data_out, index=False)
        print(f"  wrote {data_out.relative_to(REPO_ROOT)}")


def plot_train_val_overlay(
    runs_by_group: dict[tuple, list[wandb.apis.public.Run]],
    theme: str,
) -> None:
    """Overlay training loss (dotted) and validation loss (solid) per group."""
    fname = "train_val_loss_vs_iter"
    out_dir = ASSETS_DIR / theme
    fig, ax = plt.subplots(figsize=(12, 6.5))
    drew_any = False
    panel_rows: list[pd.DataFrame] = []
    for gkey, runs_in_group in runs_by_group.items():
        train_xs: list[np.ndarray] = []
        train_ys: list[np.ndarray] = []
        val_xs: list[np.ndarray] = []
        val_ys: list[np.ndarray] = []
        for run in runs_in_group:
            try:
                tdf = load_pair(run, "loss/iteration", "loss/lm loss")
                vdf = load_pair(run, "val/iteration", "val/lm loss")
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {run.id}: {e}")
                continue
            if not tdf.empty and {"loss/iteration", "loss/lm loss"}.issubset(tdf.columns):
                sub = tdf[["loss/iteration", "loss/lm loss"]].dropna()
                train_xs.append(sub["loss/iteration"].to_numpy())
                train_ys.append(sub["loss/lm loss"].to_numpy())
            if not vdf.empty and {"val/iteration", "val/lm loss"}.issubset(vdf.columns):
                sub = vdf[["val/iteration", "val/lm loss"]].dropna()
                val_xs.append(sub["val/iteration"].to_numpy())
                val_ys.append(sub["val/lm loss"].to_numpy())
        if not (train_xs or val_xs):
            continue
        dfl_short = Path(gkey[5] or "?").name
        color = None
        if train_xs:
            xt = np.concatenate(train_xs); yt = np.concatenate(train_ys)
            order = np.argsort(xt); xt, yt = xt[order], yt[order]
            # Stride down so the dotted line stays readable.
            xt, yt = xt[::10], yt[::10]
            line, = ax.plot(xt, yt, linestyle=":", linewidth=1.4,
                            label=f"{dfl_short} (train)")
            color = line.get_color()
            panel_rows.append(pd.DataFrame({
                "group_data_file": dfl_short, "kind": "train",
                "color": color, "x": xt, "y": yt,
            }))
        if val_xs:
            xv = np.concatenate(val_xs); yv = np.concatenate(val_ys)
            order = np.argsort(xv); xv, yv = xv[order], yv[order]
            line, = ax.plot(xv, yv, linestyle="-", linewidth=1.8,
                            color=color, label=f"{dfl_short} (val)")
            panel_rows.append(pd.DataFrame({
                "group_data_file": dfl_short, "kind": "val",
                "color": line.get_color(), "x": xv, "y": yv,
            }))
        drew_any = True
    if not drew_any:
        plt.close(fig); return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, handlelength=2.5,
    )
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fname}.svg"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    if theme == "light" and panel_rows:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data_out = DATA_DIR / f"{fname}.parquet"
        pd.concat(panel_rows, ignore_index=True).to_parquet(data_out, index=False)
        print(f"  wrote {data_out.relative_to(REPO_ROOT)}")


def main() -> int:
    api = wandb.Api()
    runs = fetch_runs(api)

    runs_by_group: dict[tuple, list[wandb.apis.public.Run]] = defaultdict(list)
    for r in runs:
        runs_by_group[group_key(r)].append(r)
    print(f"grouped into {len(runs_by_group)} group(s)")

    for theme in THEMES:
        print(f"theme: {theme}")
        apply_theme(theme)
        for panel in PANELS:
            print(f"  panel {panel[0]}...")
            plot_panel(runs_by_group, *panel, theme=theme)
        print(f"  panel train_val_loss_vs_iter...")
        plot_train_val_overlay(runs_by_group, theme=theme)
    return 0


if __name__ == "__main__":
    sys.exit(main())

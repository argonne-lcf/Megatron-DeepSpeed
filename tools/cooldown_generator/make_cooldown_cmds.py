#!/usr/bin/env python3

import os
from typing import Any, Optional
import argparse
from pathlib import Path
from textwrap import dedent
import ezpz

# _FILE_PATH = Path(os.path.abspath(__file__)).parent
# _MEGATRON_PATH = _FILE_PATH.parent.parent

logger = ezpz.get_logger(__name__)


def get_header_template(
    queue: str = "prod",
    project: str = "AuroraGPT",
    walltime: str = "06:00:00",
    filesystems: str = "flare:home",
    nodes: int = 256,
) -> str:
    return "\n".join(
        [
            f"#PBS -q {queue}",
            f"#PBS -A {project}",
            f"#PBS -l walltime={walltime}",
            f"#PBS -l filesystems={filesystems}",
            f"#PBS -l select={nodes}",
            "#PBS -j oe",
            "",
            "cd ${PBS_O_WORKDIR}",
            "\n",
        ]
    )


def fmt_float(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".")


def get_total_iters_from_cooldown_percent(
    checkpoint_iter: Optional[int] = None,
    cooldown_percent: Optional[float] = None,
    cooldown_steps: Optional[int] = None,
    train_iters: Optional[int] = None,
) -> dict:
    if checkpoint_iter is None and train_iters is None:
        raise ValueError("Expected one of {checkpoint_iter, train_iters}")
    if cooldown_percent is None and cooldown_steps is None:
        raise ValueError("Expected one of {cooldown_percent, cooldown_iters}")
    if checkpoint_iter is None:
        assert train_iters is not None
        if cooldown_percent is None:
            assert cooldown_steps is not None
            checkpoint_iter = train_iters - cooldown_steps
            cooldown_percent = (train_iters - cooldown_steps) / train_iters
        elif cooldown_steps is None:
            assert cooldown_percent is not None
            cooldown_steps = int(train_iters * cooldown_percent)
            checkpoint_iter = train_iters - cooldown_steps
        else:
            raise ValueError(
                "Expected one of {cooldown_percent, cooldown_iters} to be specified"
            )
        assert (
            checkpoint_iter is not None
            and cooldown_percent is not None
            and cooldown_steps is not None
            and train_iters is not None
        )
        return {
            "checkpoint_iter": checkpoint_iter,
            "cooldown_percent": cooldown_percent,
            "cooldown_iters": cooldown_steps,
            "train_iters": train_iters,
        }
    if train_iters is None:
        assert checkpoint_iter is not None
        if cooldown_percent is None:
            assert cooldown_steps is not None
            train_iters = checkpoint_iter + cooldown_steps
            cooldown_percent = (train_iters - cooldown_steps / train_iters)
        elif cooldown_steps is None:
            assert cooldown_percent is not None
            cooldown_steps = int(cooldown_percent * checkpoint_iter)
            train_iters = checkpoint_iter + cooldown_steps
        else:
            raise ValueError(
                "Expected one of {cooldown_percent, cooldown_iters}"
            )
        assert (
            checkpoint_iter is not None
            and cooldown_percent is not None
            and cooldown_steps is not None
            and train_iters is not None
        )
        return {
            "checkpoint_iter": checkpoint_iter,
            "cooldown_percent": cooldown_percent,
            "cooldown_iters": cooldown_steps,
            "train_iters": train_iters,
        }


def build_command(
    load_path: str,
    data_file_list: str,
    train_script: str,
    train_iters: Optional[int] = None,
    lr_cooldown_frac: float = 0.05,
    grad_acc_steps: Optional[int] = None,
    opt: Optional[str] = None,
    min_lr: Optional[float] = None,
    override_ckpt_opt_param: bool = True,
    extra_args: Optional[str] = None,
    model_arch: str = "AuroraGPT-2B",
    train_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    lr: Optional[float] = None,
    micro_batch: Optional[int] = None,
    use_activation_checkpointing: Optional[bool] = None,
    tokenizer_type: str = "HFTokenizer",
    tokenizer_model: str = "google/gemma-7b",
    zero_stage: Optional[str | int] = None,
) -> str:
    act_ckpt_val = "1" if use_activation_checkpointing else "0"
    override_ckpt_val = "1" if override_ckpt_opt_param else "0"
    env_lines = [
        f"MODEL_ARCH={model_arch}",
        "LR_DECAY_STYLE=constant",
        f"LOAD={load_path}",
        f"DATA_FILE_LIST={data_file_list}",
        f"USE_ACTIVATION_CHECKPOINTING={act_ckpt_val}",
        f"OVERRIDE_CKPT_OPT_PARAM={override_ckpt_val}",
        f"TOKENIZER_TYPE={tokenizer_type}",
        f"TOKENIZER_MODEL={tokenizer_model}",
    ]
    if opt is not None:
        env_lines.append(f"OPT={opt}")
    if grad_acc_steps is not None:
        env_lines.append(f"GRAD_ACC_STEPS={grad_acc_steps}")
    if lr is not None:
        env_lines.append(f"LR={lr}")
    if micro_batch is not None:
        env_lines.append(f"MICRO_BATCH={micro_batch}")
    if zero_stage is not None:
        env_lines.append(f"ZERO_STAGE={zero_stage}")

    # ---- TRAIN {ITERS, TOKENS} setup ---------------------------------------
    if train_iters is None and train_tokens is None:
        raise ValueError("One of {train_iters, train_tokens} required!")
    if train_iters is not None:
        assert train_tokens is None, (
            f"Only one of {train_tokens, train_iters} should be specified."
        )
    if train_tokens is not None:
        assert train_iters is None, (
            f"Only one of {train_tokens, train_iters} should be specified."
        )
        assert global_batch_size is not None and sequence_length is not None
        train_iters = train_tokens * global_batch_size * sequence_length

    assert train_iters is not None
    env_lines.append(f"TRAIN_ITERS={train_iters}")

    env_block = " \\\n".join([line for line in env_lines if line])

    extra_line = ""
    if extra_args:
        extra_line = f" \\\n      {extra_args}"

    cmd = dedent(f"""\
    {env_block} \\
    bash {train_script} \\
      --override-opt_param-scheduler \\
      --min-lr={min_lr} \\
      --lr_constant_plus_cooldown \\
      --lr_constant_plus_cooldown_frac={fmt_float(lr_cooldown_frac)}{extra_line}
    """).strip()
    return cmd


def parse_pairs(pairs_args):
    records = []
    next_id = 1
    for item in pairs_args:
        parts = item.split(":")
        if len(parts) == 2:
            S = int(parts[0])
            R = int(parts[1])
            cid = next_id
            next_id += 1
        elif len(parts) == 3:
            cid = int(parts[0])
            S = int(parts[1])
            R = int(parts[2])
        else:
            raise SystemExit(f"Bad --pairs entry: {item}")
        if S <= 0 or R <= 0:
            raise SystemExit(f"Non-positive S/R in --pairs entry: {item}")
        records.append({"id": cid, "S": S, "R": R})
    return records


def main():
    p = argparse.ArgumentParser(
        description="Emit Megatron-DeepSpeed cooldown commands so LR cooldown starts at resume.\n"
        "Provide checkpoint iteration(s) S and cooldown step(s) R.\n"
        "For each pair, sets TRAIN_ITERS T=S+R and lr_constant_plus_cooldown_frac f=S/T."
    )
    p.add_argument("--load", required=True)
    p.add_argument("--data-file-list", required=True)
    p.add_argument("--train-script", default="train_alcf.sh")
    p.add_argument("--grad-acc-steps", type=int, default=2)
    p.add_argument("--opt", default="ipex.fusedlamb")
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--no-override-ckpt-opt", action="store_true")
    p.add_argument("--extra-args", default="")
    p.add_argument("--emit-sh", action="store_true", default=None)
    p.add_argument("--split-by-id", action="store_true")
    p.add_argument("--queue", default="prod", type=str)
    p.add_argument("--project", default="AuroraGPT", type=str)
    p.add_argument("--walltime", default="06:00:00", type=str)
    p.add_argument("--filesystems", default="flare:home", type=str)
    p.add_argument("--nodes", default=256, type=int)
    p.add_argument("--include-header", type=str, default=None)
    p.add_argument("--train-iters", type=int, required=False)
    p.add_argument("--train-tokens", type=int, required=False)
    p.add_argument("--global-batch", type=int, required=False)
    p.add_argument("--sequence-length", type=int, required=False)
    p.add_argument("--lr", type=float, required=False)
    p.add_argument("--micro-batch", type=int, required=False)
    p.add_argument("--use-activation-checkpointing", type=int, required=False)
    p.add_argument("--tokenizer-type", type=str, required=False)
    p.add_argument("--tokenizer-model", type=str, required=False)
    p.add_argument("--zero-stage", type=int, required=False)
    p.add_argument("--checkpoint-iters", "-S", type=int, nargs="+")
    p.add_argument("--cooldown-steps", "-R", type=int, nargs="+")
    p.add_argument("--cooldown-percent", type=float, required=False)
    p.add_argument("--checkpoint-ids", type=int, nargs="+")
    p.add_argument("--pairs", type=str, nargs="*")
    args = p.parse_args()
    if args.include_header is None:
        args.include_header = get_header_template(
            queue=args.queue,
            project=args.project,
            walltime=args.walltime,
            filesystems=args.filesystems,
            nodes=args.nodes,
        )
    override_flag = not args.no_override_ckpt_opt

    if args.pairs:
        records = parse_pairs(args.pairs)
    else:
        ids = args.checkpoint_ids or list(range(1, len(args.checkpoint_iters) + 1))
        if len(ids) != len(args.checkpoint_iters):
            raise SystemExit(
                "--checkpoint-ids must match length of --checkpoint-iters."
            )
        records = [
            {"id": cid, "S": int(S), "R": int(R)}
            for cid, S in zip(ids, args.checkpoint_iters)
            for R in args.cooldown_steps
        ]

    lines = []
    # header = "# Auto-generated cooldown commands\nset -euo pipefail\n\n"
    if args.include_header:
        if (hfp := Path(args.include_header)).is_file():
            with hfp.open("r") as f:
                lines.extend(f.readlines())
        else:
            lines.extend("\n".join(args.include_header.split("\n")))
    # if args.emit_sh:

    for rec in records:
        cid, S, R = rec["id"], rec["S"], rec["R"]
        T = S + R
        f = S / T
        tag = f"# id={cid} resume_step={S} cooldown_steps={R} total_iters={T} frac={fmt_float(f)}"
        cmd = build_command(
            load_path=args.load,
            data_file_list=args.data_file_list,
            train_script=args.train_script,
            train_iters=T,
            lr_cooldown_frac=f,
            grad_acc_steps=args.grad_acc_steps,
            opt=args.opt,
            min_lr=args.min_lr,
            override_ckpt_opt_param=override_flag,
            extra_args=args.extra_args.strip(),
        )
        latest_fp = Path(args.load).parent.joinpath("latest")
        latest_ckpt_iter = Path(args.load).parent.joinpath("latest_checkpointed_iteration.txt")
        ckpt_parent = Path(args.load).parent
        if latest_fp.is_file():
            logger.info(f"Found 'latest' in {ckpt_parent}!")
            with latest_fp.open("r") as f:
                _latest = f.read().rstrip("\n").lstrip("global_step")
            assert int(_latest) == int(S), f"{_latest=} != {S=}"
        else:
            logger.info(f"No 'latest' in {ckpt_parent}!")
            logger.info(f"Writing global_step{S} to {latest_fp}")
            with latest_fp.open("w") as f:
                f.write(f"global_step{S}")

        if latest_ckpt_iter.is_file():
            logger.info(f"Found 'latest_checkpointed_iteration.txt' in {ckpt_parent}!")
            with latest_ckpt_iter.open("r") as f:
                _latest = f.read().rstrip("\n")
            assert int(_latest) == int(S), f"{_latest=} != {S=}"
        else:
            logger.info(f"No 'latest_checkpointed_iteration.txt' in {ckpt_parent}!")
            logger.info(f"Writing {S} to {latest_ckpt_iter}")
            with latest_ckpt_iter.open("w") as f:
                f.write(f"{S}")

        block = f"{tag}\n{cmd}\n"
        if args.emit_sh:
            outfile = f"cooldown_id{cid}_s{S}_r{R}_t{T}.sh"
            logger.info(f"Writing:\n{block}\nto:\n{outfile}")
            with open(outfile, "w") as f:
                f.write("".join(lines))
                f.writelines(block + "\n")
        else:
            print(block + "\n")
            # lines.append(block + "\n")

    # if args.emit_sh:
    #     for rec in records:
    #         cid, S, R = rec["id"], rec["S"], rec["R"]
    #
    #         args.emit_sh.write_text("\n".join(lines))
    # print(f"# Wrote script to: {args.emit_sh}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from typing import Any, Optional
import argparse
from pathlib import Path
from textwrap import dedent


#!/bin/bash --login
# PBS -q prod
# PBS -A AuroraGPT
# PBS -l walltime=06:00:00
# PBS -l filesystems=flare:home
# PBS -l select=256
# PBS -j oe


_FILE_PATH = Path(os.path.abspath(__file__)).parent
_MEGATRON_PATH = _FILE_PATH.parent.parent


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
        ]
    )


def fmt_float(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".")


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
    # p.add_argument()
    # p.add_argument("--include-header", default="")
    # p.add_argument("--include-header", default=None, type=)
    p.add_argument("--checkpoint-iters", "-S", type=int, nargs="+")
    p.add_argument("--cooldown-steps", "-R", type=int, nargs="+")
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
        if not args.checkpoint_iters or not args.cooldown_steps:
            raise SystemExit(
                "Provide either --pairs OR both --checkpoint-iters and --cooldown-steps."
            )
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
        block = f"{tag}\n{cmd}\n"
        if args.emit_sh:
            outfile = f"cooldown_{cid}.sh"
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

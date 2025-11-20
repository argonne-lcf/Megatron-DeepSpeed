#!/usr/bin/env python3
import argparse
from pathlib import Path
from textwrap import dedent

def fmt_float(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".")

def build_command(
    load_path: str,
    data_file_list: str,
    train_script: str,
    train_iters: int,
    lr_cooldown_frac: float,
    grad_acc_steps: int,
    opt: str,
    min_lr: float,
    override_ckpt_opt_param: bool,
    extra_args: str,
) -> str:
    env_lines = [
        "LR_DECAY_STYLE=constant",
        f"OPT={opt}",
        "OVERRIDE_CKPT_OPT_PARAM=1" if override_ckpt_opt_param else "",
        f"TRAIN_ITERS={train_iters}",
        f"GRAD_ACC_STEPS={grad_acc_steps}",
        f"LOAD={load_path}",
        f"DATA_FILE_LIST={data_file_list}",
    ]
    env_block = " \\\n".join([l for l in env_lines if l])

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
            S = int(parts[0]); R = int(parts[1]); cid = next_id; next_id += 1
        elif len(parts) == 3:
            cid = int(parts[0]); S = int(parts[1]); R = int(parts[2])
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
    p.add_argument("--emit-sh", type=Path, default=None)

    p.add_argument("--checkpoint-iters", "-S", type=int, nargs="+")
    p.add_argument("--cooldown-steps", "-R", type=int, nargs="+")
    p.add_argument("--checkpoint-ids", type=int, nargs="+")
    p.add_argument("--pairs", type=str, nargs="*")

    args = p.parse_args()
    override_flag = not args.no_override_ckpt_opt

    if args.pairs:
        records = parse_pairs(args.pairs)
    else:
        if not args.checkpoint_iters or not args.cooldown_steps:
            raise SystemExit("Provide either --pairs OR both --checkpoint-iters and --cooldown-steps.")
        ids = args.checkpoint_ids or list(range(1, len(args.checkpoint_iters) + 1))
        if len(ids) != len(args.checkpoint_iters):
            raise SystemExit("--checkpoint-ids must match length of --checkpoint-iters.")
        records = [{"id": cid, "S": int(S), "R": int(R)}
                   for cid, S in zip(ids, args.checkpoint_iters)
                   for R in args.cooldown_steps]

    lines = []
    header = "# Auto-generated cooldown commands\nset -euo pipefail\n\n"
    if args.emit_sh:
        lines.append(header)

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
        print(block)
        if args.emit_sh:
            lines.append(block + "\n")

    if args.emit_sh:
        args.emit_sh.write_text("\n".join(lines))
        print(f"# Wrote script to: {args.emit_sh}")

if __name__ == "__main__":
    main()


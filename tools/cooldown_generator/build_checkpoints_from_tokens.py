#!/usr/bin/env python3
"""
Compute training iterations (steps) for checkpoints at 1..(ttokens-1) Trillion tokens
and their "rollback" checkpoints offset by a cooldown percentage of the FINAL total.

Output TSV columns:
  id\tsteps_mod\tsteps_rollback

Where:
  steps_mod       = rounded steps at exactly i*T tokens
  steps_rollback  = rounded steps at (steps_mod - cooldown_iters) using cooldown_iters=percent*steps_at_(ttokens-1)
"""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ttokens", type=int, default=8, help="Total token milestones (default: 8 for 0..7T).")
    p.add_argument("--tokens-per-step", type=int, default=8192*6144, help="Tokens per optimizer step (default: 8192*6144).")
    p.add_argument("--cooldown-percent", type=float, default=0.05, help="Percent of final run used for rollback offset (default: 0.05).")
    p.add_argument("--round", type=int, default=50, help="Round steps to nearest N (default: 50).")
    p.add_argument("--out", type=str, default="checkpoints.tsv", help="Output TSV path (default: checkpoints.tsv).")
    args = p.parse_args()

    ttokens = args.ttokens
    tps = args.tokens_per_step
    r = args.round
    c = args.cooldown_percent

    # Steps at each i*T (i in 0..ttokens-1)
    runs = {i: (i * 10**12) / tps for i in range(ttokens)}
    runs_mod = {k: int(round(v / r) * r) for k, v in runs.items()}

    # <cooldown_percent>% of the FINAL (ttokens-1) step count, then rounded rollback
    cooldown_iters = int(c * runs[ttokens - 1])
    runs_rollback = {k: int(round((v - cooldown_iters) / r) * r) for k, v in runs_mod.items()}

    with open(args.out, "w") as f:
        f.write("id\tsteps_mod\tsteps_rollback\n")
        for k in range(1, ttokens):   # emit 1..(ttokens-1)
            f.write(f"{k}\t{runs_mod[k]}\t{runs_rollback[k]}\n")

    print(f"Wrote TSV to {args.out}")

if __name__ == "__main__":
    main()


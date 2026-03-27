"""
eval_tronformer_dist.py — Run many episodes and report per-episode profit distribution.

Usage:
    python -u eval_tronformer_dist.py \
        --model runs/tronformer_ne24_s8_v2/best_model.pt \
        --num-runs 2000 --n-bg 24 --bg-strategy 8 --skew-bins \
        --output results/equilibrium/tronformer_ne24_s8_v2/dist_2000runs.csv
"""

import argparse
import collections
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from marketsim.agent.tronformer_agent import TRONformerPolicy, SEQ_LEN
from marketsim.wrappers.tron_env import TRONEnv

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)
SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

STRATEGIES = {
    0: {'shade': [0, 450],    'eta': 0.5},
    1: {'shade': [0, 600],    'eta': 0.5},
    2: {'shade': [90, 110],   'eta': 0.5},
    3: {'shade': [140, 160],  'eta': 0.5},
    4: {'shade': [190, 210],  'eta': 0.5},
    5: {'shade': [280, 320],  'eta': 0.5},
    6: {'shade': [380, 420],  'eta': 0.5},
    7: {'shade': [380, 420],  'eta': 1.0},
    8: {'shade': [460, 540],  'eta': 0.5},
    9: {'shade': [950, 1050], 'eta': 0.5},
}

ENV = {
    'lam': 0.005, 'mean': 1e5, 'r': 0.01, 'shock_var': 1e6,
    'pv_var': 5e6, 'q_max': 10, 'sim_time': 2000,
}
NORMALIZERS = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}


def _worker(args):
    """Run a chunk of episodes; return list of per-episode profits."""
    (weights_path, n_runs, n_bg, bg_strategy, lam_zi, n_layers,
     shade_bins, seed, lam, shock_var, pv_var) = args

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(1 << 31)))

    bg = STRATEGIES[bg_strategy]
    policy = TRONformerPolicy(input_dim=14, n_layers=n_layers, shade_bins=shade_bins)
    policy.load_state_dict(torch.load(weights_path, map_location="cpu"))
    policy.eval()

    env_kwargs = dict(
        num_background_agents=n_bg,
        sim_time=ENV["sim_time"],
        lam=lam,
        lam_zi=lam_zi,
        mean=ENV["mean"],
        r=ENV["r"],
        shock_var=shock_var,
        q_max=ENV["q_max"],
        pv_var=pv_var,
        shade=[250, 500],
        normalizers=NORMALIZERS,
        bg_strategies=[{'shade': bg['shade'], 'eta': bg['eta']}],
        warmup_fraction=0.0,
        shade_bins=shade_bins,
    )
    env = TRONEnv(**env_kwargs)

    profits = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        obs_buf: collections.deque = collections.deque(maxlen=SEQ_LEN)
        ep_reward = 0.0
        done = False
        while not done:
            obs_buf.append(obs.astype(np.float32))
            obs_t = torch.tensor(
                np.stack(list(obs_buf)), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)
            shade_idx = int(Q_shade[0].argmax().item())
            eta_idx   = int(Q_eta[0].argmax().item())
            obs, r, terminated, truncated, _ = env.step(
                np.array([shade_idx, eta_idx])
            )
            ep_reward += r
            done = terminated or truncated
        profits.append(ep_reward * NORMALIZERS["reward"])
    return profits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-runs", type=int, default=2000)
    p.add_argument("--n-bg", type=int, default=24)
    p.add_argument("--bg-strategy", type=int, default=8)
    p.add_argument("--lam-zi", type=float, default=0.02)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--skew-bins", action="store_true")
    p.add_argument("--shade-max", type=float, default=600.0,
                   help="Max shade for uniform bins (ignored when --skew-bins)")
    p.add_argument("--n-bins", type=int, default=42,
                   help="Number of uniform shade bins (ignored when --skew-bins)")
    p.add_argument("--lam-market", type=float, default=None,
                   help="Background market lambda override (default: ENV 0.005)")
    p.add_argument("--shock-var", type=float, default=None,
                   help="Fundamental shock variance override (default: ENV 1e6)")
    p.add_argument("--pv-var", type=float, default=None,
                   help="Private value variance override (default: ENV 5e6)")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: auto-generated from model path)")
    args = p.parse_args()

    if args.skew_bins:
        shade_bins = SKEWED_SHADE_BINS
    elif args.shade_max != 600.0 or args.n_bins != 42:
        shade_bins = np.linspace(0, args.shade_max, args.n_bins)
    else:
        shade_bins = UNIFORM_SHADE_BINS

    lam       = args.lam_market if args.lam_market is not None else ENV["lam"]
    shock_var = args.shock_var  if args.shock_var  is not None else ENV["shock_var"]
    pv_var    = args.pv_var     if args.pv_var     is not None else ENV["pv_var"]

    n_procs = args.processes or min(mp.cpu_count(), 8)
    chunk = args.num_runs // n_procs
    remainder = args.num_runs - chunk * n_procs

    rng = np.random.default_rng(args.seed)
    chunks = [chunk + (1 if i < remainder else 0) for i in range(n_procs)]
    seeds = rng.integers(1 << 31, size=n_procs).tolist()

    tasks = [
        (args.model, chunks[i], args.n_bg, args.bg_strategy,
         args.lam_zi, args.n_layers, shade_bins, seeds[i],
         lam, shock_var, pv_var)
        for i in range(n_procs)
    ]

    bin_desc = ("skewed[0,600]x42" if args.skew_bins
                else f"uniform[0,{args.shade_max:.0f}]x{args.n_bins}")
    print(f"Running {args.num_runs} episodes across {n_procs} processes")
    print(f"  Model: {args.model}")
    print(f"  BG: {args.n_bg}x S{args.bg_strategy} ({STRATEGIES[args.bg_strategy]})")
    print(f"  lam={lam}, shock_var={shock_var:.2e}, pv_var={pv_var:.2e}")
    print(f"  lam_zi={args.lam_zi}, n_layers={args.n_layers}, bins={bin_desc}\n")

    all_profits = []
    with mp.Pool(n_procs) as pool:
        for chunk_profits in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_procs, desc="Chunks"
        ):
            all_profits.extend(chunk_profits)

    profits = np.array(all_profits)
    n = len(profits)
    mean = profits.mean()
    std  = profits.std()
    se   = std / np.sqrt(n)
    med  = np.median(profits)
    t    = mean / se if se > 0 else float('inf')

    print(f"\n{'='*58}")
    print(f"TRONformer vs {args.n_bg}x S{args.bg_strategy} — {n} episodes")
    print(f"{'='*58}")
    print(f"  Mean       : {mean:+.2f}")
    print(f"  Std dev    : {std:.2f}")
    print(f"  Std error  : {se:.2f}")
    print(f"  t-stat     : {t:.2f}")
    print(f"  95% CI     : [{mean - 1.96*se:+.2f}, {mean + 1.96*se:+.2f}]")
    print(f"  Median     : {med:+.2f}")
    print(f"  p5 / p25   : {np.percentile(profits, 5):+.2f} / {np.percentile(profits, 25):+.2f}")
    print(f"  p75 / p95  : {np.percentile(profits, 75):+.2f} / {np.percentile(profits, 95):+.2f}")
    print(f"  Min / Max  : {profits.min():+.2f} / {profits.max():+.2f}")
    print(f"  Frac > 0   : {(profits > 0).mean()*100:.1f}%")
    print(f"{'='*58}")

    if args.output:
        out = args.output
    else:
        model_dir = os.path.dirname(args.model)
        run_name  = os.path.basename(model_dir)
        out = f"results/equilibrium/{run_name}/dist_{n}runs.csv"

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    pd.Series(profits, name="profit").to_csv(out, index=False)
    print(f"\nPer-episode profits saved to {out}")


if __name__ == "__main__":
    main()

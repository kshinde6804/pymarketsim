"""
eval_tron_dist.py — Run many episodes of the TRON (LSTM R2D2) agent and report
per-episode profit distribution vs a fixed ZI background strategy.

Usage:
    python -u eval_tron_dist.py \
        --model runs/tron_paper_envA_s42/best_model.pt \
        --num-runs 500 --n-bg 24 --bg-strategy 8 \
        --lam-market 0.0005 --lam-zi 0.0005 --seed 42 \
        --output results/equilibrium/paper_envA_s8_tron_s42_eval_s42.csv
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from marketsim.agent.tron_agent import TRONPolicy
from marketsim.wrappers.tron_env import TRONEnv

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

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)
SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])


def _worker(args):
    """Run a chunk of episodes; return list of (tron_profit, mean_bg_profit)."""
    (weights_path, n_runs, n_bg, bg_strategy, lam_zi,
     shade_bins, hidden_dim, seed, lam, shock_var, pv_var) = args

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(1 << 31)))

    bg = STRATEGIES[bg_strategy]
    policy = TRONPolicy(input_dim=14, hidden_dim=hidden_dim, shade_bins=shade_bins)
    sd = torch.load(weights_path, map_location="cpu")
    policy.load_state_dict(sd)
    policy.eval()

    env = TRONEnv(
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

    results = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        h = torch.zeros(1, 1, policy.hidden_dim)
        c = torch.zeros(1, 1, policy.hidden_dim)
        ep_reward = 0.0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                Q_shade, Q_eta, (h, c) = policy(obs_t, (h, c))
            shade_idx = int(Q_shade.argmax(dim=-1).item())
            eta_idx   = int(Q_eta.argmax(dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
            ep_reward += r
            done = terminated or truncated

        tron_profit = ep_reward * NORMALIZERS["reward"]

        # BG agent mean profit (use sim_time+1 to match terminal reward calculation)
        final_fund = env.market.fundamental.get_value_at(env.sim_time + 1)
        bg_profits = [
            agent.position * final_fund + agent.cash + agent.get_pos_value()
            for agent in env.agents.values()
        ]
        mean_bg_profit = float(np.mean(bg_profits))

        results.append((tron_profit, mean_bg_profit))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to TRONPolicy .pt state_dict")
    p.add_argument("--num-runs", type=int, default=500)
    p.add_argument("--n-bg", type=int, default=24)
    p.add_argument("--bg-strategy", type=int, default=8)
    p.add_argument("--lam-zi", type=float, default=0.005,
                   help="RL agent arrival rate (must match training lam_zi)")
    p.add_argument("--lam-market", type=float, default=None,
                   help="Background market lambda (default: same as --lam-zi)")
    p.add_argument("--shock-var", type=float, default=None,
                   help="Fundamental shock variance override (default: 1e6)")
    p.add_argument("--pv-var", type=float, default=None,
                   help="Private value variance override (default: 5e6)")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--skew-bins", action="store_true",
                   help="Use skewed shade bins")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: auto-generated from model path)")
    args = p.parse_args()

    shade_bins = SKEWED_SHADE_BINS if args.skew_bins else UNIFORM_SHADE_BINS
    lam = args.lam_market if args.lam_market is not None else args.lam_zi
    shock_var = args.shock_var if args.shock_var is not None else ENV["shock_var"]
    pv_var = args.pv_var if args.pv_var is not None else ENV["pv_var"]

    n_procs = args.processes or min(mp.cpu_count(), 8)
    chunk = args.num_runs // n_procs
    remainder = args.num_runs - chunk * n_procs

    rng = np.random.default_rng(args.seed)
    chunks = [chunk + (1 if i < remainder else 0) for i in range(n_procs)]
    seeds = rng.integers(1 << 31, size=n_procs).tolist()

    tasks = [
        (args.model, chunks[i], args.n_bg, args.bg_strategy,
         args.lam_zi, shade_bins, args.hidden_dim, seeds[i],
         lam, shock_var, pv_var)
        for i in range(n_procs)
    ]

    bin_desc = "skewed" if args.skew_bins else "uniform"
    print(f"Running {args.num_runs} episodes across {n_procs} processes")
    print(f"  Model: {args.model}")
    print(f"  BG: {args.n_bg}x S{args.bg_strategy} ({STRATEGIES[args.bg_strategy]})")
    print(f"  lam={lam}, shock_var={shock_var:.2e}, pv_var={pv_var:.2e}")
    print(f"  lam_zi={args.lam_zi}, hidden_dim={args.hidden_dim}, bins={bin_desc}\n")

    all_results = []
    with mp.Pool(n_procs) as pool:
        for chunk_results in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_procs, desc="Chunks"
        ):
            all_results.extend(chunk_results)

    tron_arr = np.array([r[0] for r in all_results])
    bg_arr   = np.array([r[1] for r in all_results])
    n = len(tron_arr)

    def stats(arr):
        m = arr.mean(); s = arr.std(); se = s / np.sqrt(n)
        return m, s, se, np.median(arr), m / se if se > 0 else float('inf')

    tm, ts, tse, tmed, tt = stats(tron_arr)
    bm, bs, bse, bmed, _  = stats(bg_arr)
    rel = (tm - bm) / abs(bm) * 100 if bm != 0 else float('nan')
    diff = tm - bm
    diff_se = np.sqrt(tse**2 + bse**2)
    t_diff = diff / diff_se if diff_se > 0 else float('inf')

    W = 62
    print(f"\n{'='*W}")
    print(f"TRON vs {args.n_bg}x S{args.bg_strategy} — {n} episodes")
    print(f"{'='*W}")
    print(f"  {'':30s} {'TRON':>12}  {'ZI S8 (mean)':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Mean profit':30s} {tm:>+12.2f}  {bm:>+12.2f}")
    print(f"  {'Std dev':30s} {ts:>12.2f}  {bs:>12.2f}")
    print(f"  {'Std error':30s} {tse:>12.2f}  {bse:>12.2f}")
    print(f"  {'Median':30s} {tmed:>+12.2f}  {bmed:>+12.2f}")
    print(f"  {'t-stat (vs 0)':30s} {tt:>12.2f}")
    print(f"  {'-'*56}")
    print(f"  {'TRON advantage':30s} {diff:>+12.2f}  ({rel:>+.1f}%)")
    print(f"  {'t-stat (TRON vs ZI S8)':30s} {t_diff:>12.2f}")
    print(f"  {'95% CI (advantage)':30s} [{diff-1.96*diff_se:+.2f}, {diff+1.96*diff_se:+.2f}]")
    print(f"{'='*W}")

    if args.output:
        out = args.output
    else:
        model_dir = os.path.dirname(args.model)
        run_name  = os.path.basename(model_dir)
        out = f"results/equilibrium/{run_name}/dist_{n}runs_s{args.bg_strategy}.csv"

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    df_out = pd.DataFrame({"tron_profit": tron_arr, "bg_mean_profit": bg_arr})
    df_out.to_csv(out, index=False)
    print(f"\nPer-episode profits saved to {out}")


if __name__ == "__main__":
    main()

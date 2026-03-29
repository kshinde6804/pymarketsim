"""
Large-Scale ZI×ZI Equilibrium Experiment
==========================================
Same logic as experiments/NE_ZI-ZI/equilibrium_experiment.py but with
2000 runs/cell (4x more) for higher statistical power.

10×10 strategy matrix:
  - 10 background strategies × 10 deviator strategies = 100 cells
  - 2000 simulation runs per cell = 200,000 total simulations
  - Common Random Numbers (CRN) within each row

Output:
  equilibrium_results_large.csv   — mean deviator profit table
  equilibrium_se_large.csv        — standard errors
"""

import argparse
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from tqdm import tqdm

from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent


# ---------------------------------------------------------------------------
# Strategies (inlined to avoid import from hyphenated package name)
# ---------------------------------------------------------------------------

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

STRATEGY_LABELS = {
    0: 'S0 [0,450] η=0.5',
    1: 'S1 [0,600] η=0.5',
    2: 'S2 [90,110] η=0.5',
    3: 'S3 [140,160] η=0.5',
    4: 'S4 [190,210] η=0.5',
    5: 'S5 [280,320] η=0.5',
    6: 'S6 [380,420] η=0.5',
    7: 'S7 [380,420] η=1.0',
    8: 'S8 [460,540] η=0.5',
    9: 'S9 [950,1050] η=0.5',
}

# ---------------------------------------------------------------------------
# Market environment (matches equilibrium_experiment.py)
# ---------------------------------------------------------------------------

ENV = {
    'lam':       0.005,
    'mean':      1e5,
    'r':         0.01,
    'shock_var': 1e6,
    'pv_var':    5e6,
    'q_max':     10,
    'sim_time':  2000,
    'n_bg':      24,
}

NUM_RUNS  = 2000   # 4× the original 500 for higher statistical power
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_cell(args):
    """
    Run NUM_RUNS simulations for one (bg_idx, dev_idx) cell.

    CRN: seed = base_seed + bg_idx * 100_000 + run_idx
    dev_idx is excluded from seed so all deviators in the same row share
    the same fundamental path and background-agent arrivals.
    """
    bg_idx, dev_idx, num_runs, base_seed = args
    bg  = STRATEGIES[bg_idx]
    dev = STRATEGIES[dev_idx]

    dev_profits = []

    for run_idx in range(num_runs):
        run_seed = base_seed + bg_idx * 100_000 + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        sim = Simulator(
            num_background_agents=0,
            sim_time=ENV['sim_time'],
            num_assets=1,
            lam=ENV['lam'],
            mean=ENV['mean'],
            r=ENV['r'],
            shock_var=ENV['shock_var'],
            q_max=ENV['q_max'],
            pv_var=ENV['pv_var'],
        )
        sim.agents = {}

        # Deviator (agent 0)
        sim.agents[0] = ZIAgent(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV['q_max'],
            shade=dev['shade'],
            eta=dev['eta'],
            pv_var=ENV['pv_var'],
        )

        # Background agents (1 … n_bg)
        for i in range(1, ENV['n_bg'] + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i,
                market=sim.markets[0],
                q_max=ENV['q_max'],
                shade=bg['shade'],
                eta=bg['eta'],
                pv_var=ENV['pv_var'],
            )

        sim.run()

        fv = sim.markets[0].get_final_fundamental()

        def profit(agent):
            return agent.get_pos_value() + agent.position * fv + agent.cash

        dev_profits.append(profit(sim.agents[0]))

    return bg_idx, dev_idx, float(np.mean(dev_profits)), float(np.std(dev_profits, ddof=1))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_equilibrium_experiment(num_runs=NUM_RUNS, n_processes=None, base_seed=BASE_SEED):
    n = len(STRATEGIES)
    tasks = [(bg, dev, num_runs, base_seed) for bg in range(n) for dev in range(n)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(tasks))

    advantage     = np.zeros((n, n))
    advantage_std = np.zeros((n, n))

    total_sims = len(tasks) * num_runs
    print(
        f"\nLARGE-SCALE ZI×ZI EQUILIBRIUM EXPERIMENT"
        f"\n  Strategy matrix : {n}×{n} cells"
        f"\n  Runs per cell   : {num_runs:,}"
        f"\n  Total sims      : {total_sims:,}"
        f"\n  Processes       : {n_processes}"
        f"\n  base_seed       : {base_seed}"
        f"\n"
    )

    with mp.Pool(n_processes) as pool:
        for bg_idx, dev_idx, dev_mean, dev_std in tqdm(
            pool.imap_unordered(_run_cell, tasks),
            total=len(tasks),
            desc="cells completed",
        ):
            advantage[bg_idx, dev_idx]     = dev_mean
            advantage_std[bg_idx, dev_idx] = dev_std

    labels  = [f"S{i}" for i in range(n)]
    index   = pd.Index(labels, name="BG Strategy →")
    columns = pd.Index(labels, name="Deviator ↓")
    df    = pd.DataFrame(advantage,                          index=index, columns=columns)
    df_se = pd.DataFrame(advantage_std / np.sqrt(num_runs), index=index, columns=columns)
    return df, df_se


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(df, df_se=None):
    n = len(df)

    print("\n" + "=" * 80)
    print("DEVIATOR PROFIT TABLE  (large-scale run)")
    print("Rows = background strategy  |  Cols = deviator strategy")
    print("Value = mean deviator profit")
    print("=" * 80)
    print(df.round(1).to_string())

    best_dev = df.idxmax(axis=1)
    best_adv = df.max(axis=1)

    print("\n" + "=" * 80)
    print("BEST DEVIATOR STRATEGY FOR EACH BACKGROUND STRATEGY")
    print("=" * 80)
    header = f"{'BG Strategy':>12} │ {'Best Deviation':>14} │ {'Advantage':>12}"
    if df_se is not None:
        header += f" │ {'±SE':>10}"
    header += " │ NE?"
    print(header)
    print("─" * (len(header) + 2))

    ne_candidates = []
    for bg_label in df.index:
        best   = best_dev[bg_label]
        adv    = best_adv[bg_label]
        is_ne  = (bg_label == best)

        if is_ne:
            ne_candidates.append(bg_label)

        row = f"{bg_label:>12} │ {best:>14} │ {adv:>12.1f}"
        if df_se is not None:
            se_val = df_se.loc[bg_label, best]
            row += f" │ {se_val:>10.1f}"
        row += ("  ✓ NE" if is_ne else "")
        print(row)

    # ── Also report near-NE strategies (within 1 SE of diagonal) ──────────
    if df_se is not None:
        print("\n" + "─" * 80)
        print("NEAR-NE CHECK: strategies where diagonal profit is within 1 SE of best")
        for bg_label in df.index:
            bg_idx   = int(bg_label[1:])
            diag_val = df.loc[bg_label, bg_label]
            diag_se  = df_se.loc[bg_label, bg_label]
            best_val = best_adv[bg_label]
            best_lbl = best_dev[bg_label]
            gap      = best_val - diag_val
            threshold = df_se.loc[bg_label, best_lbl]
            near_ne  = (gap <= threshold)
            print(f"  {bg_label:>4}  diag={diag_val:>10.1f}  best={best_val:>10.1f}  "
                  f"gap={gap:>8.1f}  1SE={threshold:>8.1f}  {'≈NE' if near_ne else ''}")

    print("\n" + "=" * 80)
    print("NASH EQUILIBRIUM CANDIDATES")
    print("=" * 80)
    if ne_candidates:
        for label in ne_candidates:
            idx = int(label[1:])
            s   = STRATEGIES[idx]
            print(f"  {label}  shade={s['shade']}  eta={s['eta']}")
            print(f"       Best deviation when facing {label} = {label}"
                  f"  (advantage ≈ {best_adv[label]:.1f})")
    else:
        print("  No pure-strategy Nash equilibrium found among the 10 strategies.")
    print("=" * 80)

    return best_dev, ne_candidates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-runs", type=int, default=NUM_RUNS,
                   help=f"Simulation runs per cell (default: {NUM_RUNS})")
    p.add_argument("--processes", type=int, default=None,
                   help="Worker processes (default: cpu_count)")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base random seed (default: {BASE_SEED})")
    p.add_argument("--output", type=str, default="equilibrium_results_24bg.csv",
                   help="Output CSV for mean profits")
    p.add_argument("--output-se", type=str, default="equilibrium_se_24bg.csv",
                   help="Output CSV for standard errors")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df, df_se = run_equilibrium_experiment(
        num_runs=args.num_runs,
        n_processes=args.processes,
        base_seed=args.seed,
    )
    best_dev, ne_candidates = display_results(df, df_se)

    df.to_csv(args.output)
    print(f"\nAdvantage matrix  → {args.output}")

    df_se.to_csv(args.output_se)
    print(f"Standard errors   → {args.output_se}")

    if ne_candidates:
        print(f"\nFound NE candidates: {ne_candidates}")
        print("Run Phase 2 with:")
        for label in ne_candidates:
            idx = int(label[1:])
            print(f"  python train_mlp_ne.py --ne-strategy {idx}")
    else:
        print("\nNo pure NE found. Consider running Phase 2 with the strategy")
        print("that has the smallest gap between diagonal and best deviation.")

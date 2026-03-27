"""
eval_paper.py — Evaluate TRON vs ZI average profit (paper Table 3 replication)
================================================================================
Runs raw Simulator instances with 1 TRONAgent + 24 ZIAgents all playing the
Env-B Nash Equilibrium strategy (S8: shade=[460,540], eta=0.5).

Reports absolute profit per episode for both agent types, matching the format
of Table 3 in Mascioli et al. ICAIF 2024:
    Env B  (lambda=0.005, shock_var=1e6, pv_var=5e6):  ZI=152, TRON=170

Usage
-----
    # Quick smoke test (10 runs)
    python eval_paper.py --model runs/tron_paper_envB/best_model.pt --num-runs 10

    # Full evaluation (500 runs, parallel)
    caffeinate -disu python -u eval_paper.py \\
        --model runs/tron_paper_envB/best_model.pt \\
        --num-runs 500 \\
        --output results/equilibrium/paper_envB_eval.csv
"""

import argparse
import math
import multiprocessing as mp
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from marketsim.agent.tron_agent import TRONAgent
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Environment and strategy constants (Env B — matches train_tron.py defaults)
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

NORMALIZERS = {"fundamental": 1e5, "invt": 10, "pv": 5e5}

DEFAULT_MODEL    = "runs/tron_paper_envB/best_model.pt"
DEFAULT_RUNS     = 500
DEFAULT_BG_STRAT = 8   # S8 = NE for Env B with n_bg=24
BASE_SEED        = 7777


# ---------------------------------------------------------------------------
# Worker: one simulation run with TRONAgent + n_bg ZIAgents
# ---------------------------------------------------------------------------

def _run_one(args):
    """Run a single simulation and return (tron_profit, mean_zi_profit).

    Agent layout:
        Agent 0        -> TRONAgent   (loads weights from weights_path)
        Agents 1..n_bg -> ZIAgent     (all play bg strategy)

    Profits are computed as: get_pos_value() + position * final_fundamental + cash
    This equals the agent's net mark-to-market PnL, same definition as the paper.

    Returns:
        (tron_profit, mean_zi_profit)  both in raw dollar units
    """
    weights_path, bg_idx, run_idx, base_seed, n_bg = args

    run_seed = base_seed + run_idx
    random.seed(run_seed)
    np.random.seed(run_seed)

    bg = STRATEGIES[bg_idx]

    # ── Build simulator ────────────────────────────────────────────────────
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

    # ── Agent 0: TRONAgent ────────────────────────────────────────────────
    tron = TRONAgent(
        agent_id=0,
        market=sim.markets[0],
        q_max=ENV['q_max'],
        pv_var=ENV['pv_var'],
        shade=[250, 500],
        normalizers=NORMALIZERS,
        weights_path=weights_path,
    )
    tron.reset()  # zero LSTM state, reset position/cash/pv
    sim.agents[0] = tron

    # ── Agents 1..n_bg: ZIAgent at bg strategy ────────────────────────────
    for i in range(1, n_bg + 1):
        sim.agents[i] = ZIAgent(
            agent_id=i,
            market=sim.markets[0],
            q_max=ENV['q_max'],
            shade=bg['shade'],
            eta=bg['eta'],
            pv_var=ENV['pv_var'],
        )

    # Reseed event queue for reproducible order shuffling
    sim.markets[0].event_queue.rand = random.Random(run_seed + 1)

    sim.run()

    fv = sim.markets[0].get_final_fundamental()

    def profit(agent):
        return agent.get_pos_value() + agent.position * fv + agent.cash

    tron_profit = profit(sim.agents[0])
    zi_profits  = [profit(sim.agents[i]) for i in range(1, n_bg + 1)]

    return float(tron_profit), float(np.mean(zi_profits))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    weights_path: str,
    num_runs: int = DEFAULT_RUNS,
    bg_idx: int = DEFAULT_BG_STRAT,
    n_processes: int = None,
) -> tuple:
    """Run num_runs simulations and return (tron_profits, zi_profits) arrays."""
    if n_processes is None:
        n_processes = min(mp.cpu_count(), num_runs)

    tasks = [(weights_path, bg_idx, i, BASE_SEED, ENV['n_bg']) for i in range(num_runs)]

    tron_profits = np.zeros(num_runs)
    zi_profits   = np.zeros(num_runs)

    print(f"\nEvaluating TRON vs ZI (Env B)")
    print(f"  Model:       {weights_path}")
    bg = STRATEGIES[bg_idx]
    print(f"  BG strategy: S{bg_idx} (shade={bg['shade']}, eta={bg['eta']})")
    print(f"  n_bg:        {ENV['n_bg']}  (N={ENV['n_bg']+1} total agents)")
    print(f"  Runs:        {num_runs}  |  Processes: {n_processes}")
    print(f"  ENV:  lam={ENV['lam']}, shock_var={ENV['shock_var']}, pv_var={ENV['pv_var']}\n")

    with mp.Pool(n_processes) as pool:
        for i, (tp, zp) in enumerate(tqdm(
            pool.imap(_run_one, tasks),
            total=num_runs,
            desc="simulations",
        )):
            tron_profits[i] = tp
            zi_profits[i]   = zp

    return tron_profits, zi_profits


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(tron_profits: np.ndarray, zi_profits: np.ndarray, num_runs: int):
    n = len(tron_profits)
    se = lambda arr: arr.std(ddof=1) / math.sqrt(n)

    tron_mean = tron_profits.mean()
    tron_se   = se(tron_profits)
    zi_mean   = zi_profits.mean()
    zi_se     = se(zi_profits)

    pct = (tron_mean - zi_mean) / abs(zi_mean) * 100 if zi_mean != 0 else float('nan')

    print("\n" + "=" * 60)
    print("RESULTS  —  Environment B  (lam=0.005, shock_var=1e6, pv_var=5e6)")
    print(f"NE Background: S{DEFAULT_BG_STRAT} (shade=[460,540], eta=0.5), n_bg={ENV['n_bg']}")
    print(f"Runs: {num_runs}")
    print("=" * 60)
    print(f"  ZI  avg profit:  {zi_mean:>8.1f}  ±{zi_se:.1f}")
    print(f"  TRON profit:     {tron_mean:>8.1f}  ±{tron_se:.1f}")
    print(f"  TRON advantage:  {pct:>+.1f}%")
    print("-" * 60)
    print(f"  Paper (Table 3, Env B):  ZI=152, TRON=170  (+12%)")
    print("=" * 60)

    return {
        'tron_mean': tron_mean, 'tron_se': tron_se,
        'zi_mean':   zi_mean,   'zi_se':   zi_se,
        'pct_advantage': pct,
        'n_runs': num_runs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TRON vs ZI average profit (paper Table 3 replication)"
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Path to TRONPolicy .pt state_dict (default: {DEFAULT_MODEL})")
    p.add_argument("--num-runs", type=int, default=DEFAULT_RUNS,
                   help=f"Number of simulation runs (default: {DEFAULT_RUNS})")
    p.add_argument("--bg-strategy", type=int, default=DEFAULT_BG_STRAT,
                   help=f"ZI background strategy index 0-9 (default: {DEFAULT_BG_STRAT} = S8)")
    p.add_argument("--processes", type=int, default=None,
                   help="Worker processes (default: cpu_count)")
    p.add_argument("--output", default="results/equilibrium/paper_envB_eval.csv",
                   help="Output CSV path (default: results/equilibrium/paper_envB_eval.csv)")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base random seed (default: {BASE_SEED})")
    p.add_argument("--lam", type=float, default=None,
                   help="Override market arrival rate lambda (default: ENV default 0.005)")
    p.add_argument("--shock-var", type=float, default=None,
                   help="Override fundamental shock variance (default: ENV default 1e6)")
    p.add_argument("--pv-var", type=float, default=None,
                   help="Override private value variance (default: ENV default 5e6)")
    return p.parse_args()


def main():
    args = parse_args()

    global BASE_SEED
    BASE_SEED = args.seed

    if args.lam is not None:
        ENV['lam'] = args.lam
    if args.shock_var is not None:
        ENV['shock_var'] = args.shock_var
    if args.pv_var is not None:
        ENV['pv_var'] = args.pv_var

    tron_profits, zi_profits = run_evaluation(
        weights_path=args.model,
        num_runs=args.num_runs,
        bg_idx=args.bg_strategy,
        n_processes=args.processes,
    )

    summary = display_results(tron_profits, zi_profits, args.num_runs)

    # Save per-run results
    df = pd.DataFrame({
        'run': range(args.num_runs),
        'tron_profit': tron_profits,
        'zi_mean_profit': zi_profits,
    })
    df.to_csv(args.output, index=False)
    print(f"\nPer-run results saved to {args.output}")

    # Save summary
    summary_path = args.output.replace('.csv', '_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

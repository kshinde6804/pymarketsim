"""
Equilibrium Experiment
======================
For each candidate background strategy (one of 10), place 15 background agents
on that strategy and sweep a single deviator agent across all 10 strategies.
Record the deviator's profit advantage (deviator profit − mean background profit)
averaged over NUM_RUNS simulation runs.

A strategy S* is a Nash equilibrium candidate if the deviator's best response
when facing 15 agents on S* is S* itself (no profitable deviation exists).

Output: 10×10 advantage table + best-deviation summary.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from marketsim.simulator.simulator import Simulator  # direct submodule import avoids hbl_agent dep
from marketsim.agent.zero_intelligence_agent import ZIAgent


# ---------------------------------------------------------------------------
# Strategies
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
# Market environment (Environment B from experiment_framework.py)
# ---------------------------------------------------------------------------

ENV = {
    'lam':       0.005,   # arrival intensity
    'mean':      1e5,     # long-run fundamental value
    'r':         0.01,    # mean-reversion speed (kappa)
    'shock_var': 1e6,     # fundamental shock variance
    'pv_var':    5e6,     # private value variance
    'q_max':     10,      # max agent position
    'sim_time':  1000,    # time steps per simulation run
    'n_bg':      15,      # number of background agents
}

NUM_RUNS = 100  # repetitions per (bg_strategy, dev_strategy) cell


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _run_cell(args):
    """
    Run NUM_RUNS simulations for one (bg_idx, dev_idx) cell.

    Agent layout:
        Agent 0         → deviator (plays dev strategy)
        Agents 1 .. n_bg → background (all play bg strategy)

    Returns:
        (bg_idx, dev_idx, mean_dev_profit, mean_bg_profit)
    """
    bg_idx, dev_idx, num_runs = args
    bg = STRATEGIES[bg_idx]
    dev = STRATEGIES[dev_idx]

    dev_profits, bg_profits = [], []

    for _ in range(num_runs):
        sim = Simulator(
            num_background_agents=0,  # we populate agents manually
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

        # Deviator
        sim.agents[0] = ZIAgent(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV['q_max'],
            shade=dev['shade'],
            eta=dev['eta'],
            pv_var=ENV['pv_var'],
        )

        # Background agents
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
        bg_profits.append(
            float(np.mean([profit(sim.agents[i]) for i in range(1, ENV['n_bg'] + 1)]))
        )

    return bg_idx, dev_idx, float(np.mean(dev_profits)), float(np.mean(bg_profits))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_equilibrium_experiment(num_runs=NUM_RUNS, n_processes=None):
    """
    Run the full 10×10 strategy matrix experiment.

    Returns:
        advantage_df  : DataFrame of shape (10, 10)
                        advantage_df[bg][dev] = mean(dev_profit) − mean(bg_profit)
    """
    n = len(STRATEGIES)
    tasks = [(bg, dev, num_runs) for bg in range(n) for dev in range(n)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(tasks))

    advantage = np.zeros((n, n))

    total_sims = len(tasks) * num_runs
    print(
        f"Strategy matrix: {n}×{n} cells  |  {num_runs} runs/cell  |  "
        f"{total_sims:,} total simulations  |  {n_processes} processes"
    )

    with mp.Pool(n_processes) as pool:
        for bg_idx, dev_idx, dev_mean, bg_mean in tqdm(
            pool.imap_unordered(_run_cell, tasks),
            total=len(tasks),
            desc="cells completed",
        ):
            advantage[bg_idx, dev_idx] = dev_mean - bg_mean

    labels = [f"S{i}" for i in range(n)]
    df = pd.DataFrame(
        advantage,
        index=pd.Index(labels, name="BG Strategy →"),
        columns=pd.Index(labels, name="Deviator ↓"),
    )
    return df


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(df):
    # ── Advantage matrix ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROFIT ADVANTAGE TABLE")
    print("Rows = background strategy (what the 15 BG agents play)")
    print("Cols = deviator strategy   (what the 1 test agent plays)")
    print("Value = mean(deviator profit) − mean(background profit)")
    print("=" * 80)
    print(df.round(1).to_string())

    # ── Best deviator per row ───────────────────────────────────────────────
    best_dev = df.idxmax(axis=1)
    best_adv = df.max(axis=1)

    print("\n" + "=" * 80)
    print("BEST DEVIATOR STRATEGY FOR EACH BACKGROUND STRATEGY")
    print("=" * 80)
    print(f"{'BG Strategy':>12} │ {'Best Deviation':>14} │ {'Advantage':>10} │ NE?")
    print("─" * 52)
    ne_candidates = []
    for bg_label in df.index:
        best = best_dev[bg_label]
        adv = best_adv[bg_label]
        is_ne = (bg_label == best)
        ne_str = " ✓ NE" if is_ne else ""
        if is_ne:
            ne_candidates.append(bg_label)
        print(f"{bg_label:>12} │ {best:>14} │ {adv:>10.1f}{ne_str}")

    # ── Equilibrium summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NASH EQUILIBRIUM CANDIDATES")
    print("=" * 80)
    if ne_candidates:
        for label in ne_candidates:
            idx = int(label[1:])
            s = STRATEGIES[idx]
            print(f"  {label}  shade={s['shade']}  eta={s['eta']}")
            print(f"       Best deviation when facing this strategy = {label} "
                  f"(advantage ≈ {best_adv[label]:.1f})")
    else:
        print("  No pure-strategy Nash equilibrium found among the 10 strategies.")
    print("=" * 80)

    return best_dev


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_equilibrium_experiment()
    best_dev = display_results(df)

    out_file = "equilibrium_results.csv"
    df.to_csv(out_file)
    print(f"\nAdvantage matrix saved to {out_file}")

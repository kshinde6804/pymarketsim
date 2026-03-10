"""
Equilibrium Experiment — with TRON Recurrent DQN Deviator
==========================================================
Extends the 10×10 ZI strategy matrix from equilibrium_experiment.py to a
10×11 matrix by adding the TRON-trained agent as an additional deviator.

For each of the 10 background ZI strategies, the TRON agent competes as the
deviator alongside the 10 existing ZI deviator strategies.  The experiment
loads pre-computed ZI×ZI results from equilibrium_results.csv and only runs
the 10 new TRON-deviator cells.

Usage
-----
    python equilibrium_tron.py
    python equilibrium_tron.py --model runs/tron_v1/best_model.pt
    python equilibrium_tron.py --num-runs 500 --processes 4
    python equilibrium_tron.py --output equilibrium_tron_results.csv
"""

import argparse

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.tron_agent import TRONAgent

# ---------------------------------------------------------------------------
# Strategy / environment constants (inlined from equilibrium_experiment.py)
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

ENV = {
    'lam':       0.005,
    'mean':      1e5,
    'r':         0.01,
    'shock_var': 1e6,
    'pv_var':    5e6,
    'q_max':     10,
    'sim_time':  2000,
    'n_bg':      15,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "runs/tron_v1/best_model.pt"
DEFAULT_RUNS  = 1000
TRON_LABEL    = "TRON-R2D2"
NORMALIZERS   = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}
BASELINE_CSV  = "equilibrium_results.csv"


# ---------------------------------------------------------------------------
# TRONDeviator — thin subclass that loads weights and uses TRONAgent.take_action()
# ---------------------------------------------------------------------------

class TRONDeviator(TRONAgent):
    """TRON agent pre-loaded with trained weights for use as a deviator.

    Inherits TRONAgent entirely — take_action() and build_obs() are unchanged.
    Weights are loaded from weights_path at construction time.

    Args:
        agent_id:     Unique integer identifier.
        market:       The shared Market object.
        q_max:        Maximum absolute inventory position.
        pv_var:       Variance of private value draws.
        normalizers:  Feature normalisation dict.
        weights_path: Path to TRONPolicy state_dict (.pt).
    """

    def __init__(
        self,
        agent_id: int,
        market,
        q_max: int,
        pv_var: float,
        normalizers: dict,
        weights_path: str,
    ):
        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            pv_var=pv_var,
            shade=[250, 500],   # dummy — not used in take_action()
            normalizers=normalizers,
            weights_path=weights_path,
        )


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess — loads policy locally)
# ---------------------------------------------------------------------------

def _run_tron_cell(args):
    """Run num_runs simulations with TRON deviator vs one ZI background strategy.

    The policy weights are loaded inside the worker to avoid pickle/multiprocessing
    issues with PyTorch objects.

    Args:
        args: (bg_idx, weights_path, num_runs)

    Returns:
        (bg_idx, mean_deviator_profit)
    """
    bg_idx, weights_path, num_runs = args

    bg = STRATEGIES[bg_idx]
    dev_profits = []

    for _ in range(num_runs):
        sim = Simulator(
            num_background_agents=0,
            sim_time=ENV["sim_time"],
            num_assets=1,
            lam=ENV["lam"],
            mean=ENV["mean"],
            r=ENV["r"],
            shock_var=ENV["shock_var"],
            q_max=ENV["q_max"],
            pv_var=ENV["pv_var"],
        )
        sim.agents = {}

        # Deviator: TRON agent with pre-trained weights
        tron = TRONDeviator(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV["q_max"],
            pv_var=ENV["pv_var"],
            normalizers=NORMALIZERS,
            weights_path=weights_path,
        )
        sim.agents[0] = tron

        # Background agents: all play the fixed ZI strategy
        for i in range(1, ENV["n_bg"] + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i,
                market=sim.markets[0],
                q_max=ENV["q_max"],
                shade=bg["shade"],
                eta=bg["eta"],
                pv_var=ENV["pv_var"],
            )

        sim.run()

        fv  = sim.markets[0].get_final_fundamental()
        dev = sim.agents[0]
        dev_profits.append(dev.get_pos_value() + dev.position * fv + dev.cash)

        # Reset TRON LSTM state between runs (position/cash reset handled by
        # constructing a new object per run, so this is a no-op here, but
        # included for clarity if the loop is ever refactored to reuse objects)
        tron.reset()

    return bg_idx, float(np.mean(dev_profits))


# ---------------------------------------------------------------------------
# Run the TRON deviator column
# ---------------------------------------------------------------------------

def run_tron_column(
    weights_path: str,
    num_runs: int = DEFAULT_RUNS,
    n_processes: int = None,
) -> np.ndarray:
    """Run num_runs sims for each of the 10 ZI background strategies.

    Returns:
        tron_column: float array of shape (10,) — mean TRON deviator profit
                     for each ZI background strategy.
    """
    n_bg = len(STRATEGIES)
    tasks = [(bg, weights_path, num_runs) for bg in range(n_bg)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_bg)

    tron_column = np.zeros(n_bg)

    total_sims = n_bg * num_runs
    print(
        f"TRON deviator column:  {n_bg} cells  |  {num_runs} runs/cell  |  "
        f"{total_sims:,} total simulations  |  {n_processes} processes"
    )
    print(f"  Model: {weights_path}")
    print(f"  Equilibrium ENV: lam={ENV['lam']}, sim_time={ENV['sim_time']}, r={ENV['r']}.\n")

    with mp.Pool(n_processes) as pool:
        for bg_idx, tron_mean in tqdm(
            pool.imap_unordered(_run_tron_cell, tasks),
            total=n_bg,
            desc="TRON cells completed",
        ):
            tron_column[bg_idx] = tron_mean

    return tron_column


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_extended_results(
    zi_df: pd.DataFrame,
    tron_column: np.ndarray,
    model_label: str = TRON_LABEL,
) -> pd.DataFrame:
    """Display 10×11 advantage table and best-deviation analysis."""
    tron_series = pd.Series(tron_column, index=zi_df.index, name=model_label)
    df = pd.concat([zi_df, tron_series], axis=1)

    # ── Advantage matrix ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DEVIATOR PROFIT TABLE  (extended with TRON-R2D2 deviator)")
    print("Rows = background strategy  |  Cols = deviator strategy")
    print(f"Value = mean deviator profit  |  Last col ({model_label}) = TRON recurrent DQN")
    print("=" * 90)
    print(df.round(1).to_string())

    # ── Best deviator per row ─────────────────────────────────────────────
    best_dev = df.idxmax(axis=1)
    best_adv = df.max(axis=1)

    print("\n" + "=" * 90)
    print("BEST DEVIATOR STRATEGY FOR EACH BACKGROUND STRATEGY")
    print("=" * 90)
    print(
        f"{'BG Strategy':>12} │ {'Best Deviation':>14} │ {'Advantage':>10} │ NE? │ "
        f"TRON better than ZI best?"
    )
    print("─" * 75)

    ne_candidates = []
    tron_beats_zi = []

    for bg_label in df.index:
        best     = best_dev[bg_label]
        adv      = best_adv[bg_label]
        zi_best  = zi_df.idxmax(axis=1)[bg_label]
        zi_adv   = zi_df.max(axis=1)[bg_label]
        tron_val = tron_column[list(df.index).index(bg_label)]
        is_ne    = (bg_label == best)
        tron_wins = (best == model_label)

        ne_str   = " ✓" if is_ne else ""
        win_str  = (
            f"Yes (+{tron_val - zi_adv:+.0f})"
            if tron_wins
            else f"No  ({zi_best} +{zi_adv:.0f})"
        )

        if is_ne:
            ne_candidates.append(bg_label)
        if tron_wins:
            tron_beats_zi.append(bg_label)

        print(f"{bg_label:>12} │ {best:>14} │ {adv:>10.1f}{ne_str:3s} │     │ {win_str}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"  TRON is the best deviator against {len(tron_beats_zi)}/10 "
        f"background strategies:"
    )
    for bg in tron_beats_zi:
        bg_idx  = list(df.index).index(bg)
        s       = STRATEGIES[int(bg[1:])]
        zi_best = zi_df.idxmax(axis=1)[bg]
        zi_adv  = zi_df.max(axis=1)[bg]
        print(
            f"    {bg} (bg={s['shade']} η={s['eta']})  "
            f"TRON={tron_column[bg_idx]:.1f}  "
            f"best-ZI={zi_best} ({zi_adv:.1f})  "
            f"Δ={tron_column[bg_idx] - zi_adv:+.1f}"
        )

    if ne_candidates:
        print(f"\n  Nash equilibrium candidates (pure-strategy): {ne_candidates}")
    else:
        print("\n  No pure-strategy NE found among the 11 strategies.")

    print("=" * 90)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to TRONPolicy .pt state_dict (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--num-runs", type=int, default=DEFAULT_RUNS,
        help=f"Simulations per TRON-deviator cell (default: {DEFAULT_RUNS})",
    )
    p.add_argument(
        "--processes", type=int, default=None,
        help="Worker processes (default: cpu_count)",
    )
    p.add_argument(
        "--output", default="equilibrium_tron_results.csv",
        help="Output CSV path (default: equilibrium_tron_results.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ZI×ZI baseline ───────────────────────────────────────────────
    print(f"Loading ZI×ZI baseline from {BASELINE_CSV} ...")
    zi_df = pd.read_csv(BASELINE_CSV, index_col=0)
    print(f"  Loaded {zi_df.shape[0]}×{zi_df.shape[1]} matrix.\n")

    # ── Run TRON deviator column ──────────────────────────────────────────
    tron_column = run_tron_column(
        weights_path=args.model,
        num_runs=args.num_runs,
        n_processes=args.processes,
    )

    # ── Display extended results ──────────────────────────────────────────
    ext_df = display_extended_results(zi_df, tron_column)

    # ── Save ─────────────────────────────────────────────────────────────
    ext_df.to_csv(args.output)
    print(f"\nExtended advantage matrix saved to {args.output}")


if __name__ == "__main__":
    main()

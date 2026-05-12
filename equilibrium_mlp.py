"""
Equilibrium Experiment — with SAC-trained MLP Deviator
=======================================================
Extends the 10×10 ZI strategy matrix from equilibrium_experiment.py to a
10×11 matrix by adding the SAC-trained MLP agent as an additional deviator.

For each of the 10 background ZI strategies, the MLP agent competes as the
deviator alongside the 10 existing ZI deviator strategies.  The experiment
loads pre-computed ZI×ZI results from equilibrium_results.csv (if found) and
only runs the 10 new MLP-deviator cells.

Note: The MLP agent was trained in ZIEnv (lam=0.1, shock_var=5e6, sim_time=1000,
r=0.05) but is evaluated here in the equilibrium ENV (lam=0.005, shock_var=1e6,
sim_time=2000, r=0.01).  Results reflect out-of-distribution generalisation.

Usage
-----
    python equilibrium_mlp.py
    python equilibrium_mlp.py --model runs/sac_zi_v1/best_model
    python equilibrium_mlp.py --num-runs 500 --processes 4
"""

import argparse
import math
import random
from typing import List

import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.mlp_agent import MLPAgent
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.wrappers.metrics import (
    midprice_move,
    queue_imbalance,
    realized_volatility,
    relative_strength_index,
    volume_imbalance,
)

# Strategy definitions and environment config (inlined from equilibrium_experiment.py)
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
    'n_bg':      24,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "runs/sac_zi_v2/best_model"
DEFAULT_RUNS   = 1000        # runs per MLP-deviator cell
MLP_LABEL      = "MLP-SAC"
MLP_COL        = 10          # column index in the extended matrix
NORMALIZERS    = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}
SHADE_RANGE    = [0, 1000]   # must match train_mlp.py shade_range
BASELINE_CSV   = "results/equilibrium/equilibrium_results.csv"

# Market environment presets — must match train_mlp.py _ENV_PRESETS
_ENV_PRESETS = {
    'A': dict(lam=0.0005, shock_var=1e6,  pv_var=5e6),
    'B': dict(lam=0.005,  shock_var=1e6,  pv_var=5e6),
    'C': dict(lam=0.012,  shock_var=2e4,  pv_var=2e7),
}


# ---------------------------------------------------------------------------
# SACDeviator — MLPAgent driven by a pre-trained SB3 SAC policy
# ---------------------------------------------------------------------------

class SACDeviator(MLPAgent):
    """ZI-style trading agent driven by a pre-trained SB3 SAC policy.

    Inherits MLPAgent for build_obs() and position tracking.
    Overrides build_obs() to use market.end_time (correct for both
    GaussianMeanReverting and LazyGaussianMeanReverting fundamentals).
    Overrides take_action() to use sac_model.predict() instead of the
    internal ZIMlpPolicy.
    """

    def __init__(self, agent_id: int, market, q_max: int, pv_var: float,
                 shade_range: List, normalizers: dict, sac_model):
        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            pv_var=pv_var,
            shade=[250, 500],     # dummy — not used in take_action()
            shade_range=shade_range,
            normalizers=normalizers,
        )
        self.sac_model = sac_model

    # ── Observation ───────────────────────────────────────────────────────────

    def build_obs(self, side: int) -> np.ndarray:
        """14-feature observation using market.end_time as sim_time.

        Args:
            side: BUY (1) or SELL (-1) — pre-sampled before building obs so the
                  agent can condition shade/eta on which side it will trade.

        Uses market.end_time instead of fundamental.final_time - 1 so that
        time_left is computed correctly for GaussianMeanReverting (which sets
        final_time = sim_time, not sim_time+1 like LazyGaussianMeanReverting).
        """
        t = self.market.get_time()
        sim_time = self.market.end_time          # correct regardless of fundamental type

        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask = self.market.order_book.get_best_ask()
        best_bid = self.market.order_book.get_best_bid()
        inventory = self.position

        mp_delta = midprice_move(self.market)
        vol_imb  = volume_imbalance(self.market)
        que_imb  = queue_imbalance(self.market)
        rv       = realized_volatility(self.market)
        rsi      = relative_strength_index(self.market)

        est_fund = self.estimate_fundamental()
        pv_buy   = self.pv.value_for_exchange(self.position, BUY)
        pv_sell  = self.pv.value_for_exchange(self.position, SELL)

        N    = self.normalizers
        pv_n = N.get("pv", 5e5)

        time_left_n = (sim_time - t) / sim_time if sim_time > 0 else 0.0
        fund_n      = fundamental_value / N["fundamental"]
        ask_n       = 1.0 if math.isinf(best_ask) else best_ask / N["fundamental"]
        bid_n       = 0.0 if math.isinf(best_bid) else best_bid / N["fundamental"]
        inv_n       = np.clip(inventory / N["invt"], -1.0, 1.0)
        mp_n        = np.clip(mp_delta / 1e2, -1.0, 1.0)
        rsi_n       = rsi / 100.0
        est_n       = est_fund / N["fundamental"]
        pv_buy_n    = np.clip(pv_buy  / pv_n, -1.0, 1.0)
        pv_sell_n   = np.clip(pv_sell / pv_n, -1.0, 1.0)
        side_n      = 0.0 if side == BUY else 1.0

        obs = np.array(
            [time_left_n, fund_n, ask_n, bid_n, inv_n, mp_n,
             vol_imb, que_imb, np.clip(rv, 0.0, 1.0), rsi_n,
             est_n, pv_buy_n, pv_sell_n, side_n],
            dtype=np.float64,
        )
        # Guard against NaN/inf from metrics with no history (no warm-up in Simulator).
        # rsi defaults to 50 (neutral → 0.5); all others default to 0.
        if np.isnan(obs[9]):  # rsi_n
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    def take_action(self) -> List[Order]:
        """Use SAC policy to select (shade, eta), then apply ZI pricing.

        Side is pre-sampled before building the observation so that the agent
        can condition shade/eta on which side it will be trading.
        """
        side = random.choice([BUY, SELL])
        obs = self.build_obs(side)
        action, _ = self.sac_model.predict(obs, deterministic=True)
        shade_norm = float(action[0])
        eta        = float(action[1])

        d_min, d_max = self.shade_range
        shade_val = shade_norm * (d_max - d_min) + d_min

        t        = self.market.get_time()
        estimate = self.estimate_fundamental()
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - shade_val
        else:
            price = estimate + pv_value + shade_val

        if eta < 1.0:
            base_price = estimate + pv_value
            if side == BUY:
                best = self.market.order_book.get_best_ask()
                if (base_price - best) > eta * shade_val and not math.isinf(best):
                    price = best
            else:
                best = self.market.order_book.get_best_bid()
                if (best - base_price) > eta * shade_val and not math.isinf(best):
                    price = best

        self._order_counter += 1
        order_id = self.agent_id * 1000000 + self._order_counter

        return [Order(
            price=price, quantity=1, agent_id=self.agent_id,
            time=t, order_type=side, order_id=order_id,
        )]


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess — loads SAC model locally)
# ---------------------------------------------------------------------------

def _run_mlp_cell(args):
    """Run num_runs simulations with MLP deviator vs one ZI background strategy.

    The SAC model is loaded inside the worker to avoid pickle/multiprocessing
    issues with PyTorch objects.

    Returns:
        (bg_idx, mean_deviator_profit)
    """
    bg_idx, model_path, num_runs = args

    # Load SAC model inside the subprocess
    from stable_baselines3 import SAC
    sac_model = SAC.load(model_path, device="cpu")
    sac_model.policy.set_training_mode(False)

    bg = STRATEGIES[bg_idx]
    dev_profits = []

    for _ in range(num_runs):
        sim_time = ENV["sim_time"]
        n_bg     = ENV["n_bg"]
        lam      = ENV["lam"]
        q_max    = ENV["q_max"]
        pv_var   = ENV["pv_var"]

        fundamental = LazyGaussianMeanReverting(
            mean=ENV["mean"],
            final_time=sim_time + 1,
            r=ENV["r"],
            shock_var=ENV["shock_var"],
        )
        market = Market(fundamental=fundamental, time_steps=sim_time)

        deviator_id = n_bg
        all_agents = {}

        # Background agents: all play the fixed ZI strategy bg (IDs 0..n_bg-1)
        for i in range(n_bg):
            all_agents[i] = ZIAgent(
                agent_id=i, market=market, q_max=q_max,
                shade=bg["shade"], eta=bg["eta"], pv_var=pv_var,
            )

        # Deviator: MLP-SAC agent (ID n_bg)
        all_agents[deviator_id] = SACDeviator(
            agent_id=deviator_id, market=market, q_max=q_max, pv_var=pv_var,
            shade_range=SHADE_RANGE, normalizers=NORMALIZERS, sac_model=sac_model,
        )

        # Schedule initial arrivals — all agents via Geometric(lam) inter-arrivals
        arrivals = defaultdict(list)
        arr_buf = np.random.geometric(lam, 50000) - 1  # 0-indexed geometric
        arr_idx = 0
        for agent_id in all_agents:
            arrivals[int(arr_buf[arr_idx])].append(agent_id)
            arr_idx += 1

        # Simulation loop (matches ZIEnv's event-driven internals)
        for t in range(sim_time):
            if not arrivals[t]:
                continue
            market.event_queue.set_time(t)
            for agent_id in arrivals[t]:
                agent = all_agents[agent_id]
                market.withdraw_all(agent_id)
                orders = agent.take_action()
                market.add_orders(orders)
                if arr_idx >= len(arr_buf):
                    arr_buf = np.random.geometric(lam, 50000) - 1
                    arr_idx = 0
                arrivals[int(arr_buf[arr_idx]) + 1 + t].append(agent_id)
                arr_idx += 1
            new_orders = market.step()
            for matched in new_orders:
                aid  = matched.order.agent_id
                qty  = matched.order.order_type * matched.order.quantity
                cash = -matched.price * matched.order.quantity * matched.order.order_type
                all_agents[aid].update_position(qty, cash)

        market.event_queue.set_time(sim_time)
        fv  = market.get_final_fundamental()
        dev = all_agents[deviator_id]
        dev_profits.append(dev.get_pos_value() + dev.position * fv + dev.cash)

    return bg_idx, float(np.mean(dev_profits))


# ---------------------------------------------------------------------------
# Run the MLP deviator column
# ---------------------------------------------------------------------------

def run_mlp_column(model_path: str, num_runs: int = DEFAULT_RUNS,
                   n_processes: int = None) -> np.ndarray:
    """Run num_runs sims for each of the 10 ZI background strategies.

    Returns:
        mlp_column  : float array of shape (10,) — mean MLP deviator profit
                      for each ZI background strategy.
    """
    n_bg = len(STRATEGIES)
    tasks = [(bg, model_path, num_runs) for bg in range(n_bg)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_bg)

    mlp_column = np.zeros(n_bg)

    total_sims = n_bg * num_runs
    print(
        f"MLP deviator column:  {n_bg} cells  |  {num_runs} runs/cell  |  "
        f"{total_sims:,} total simulations  |  {n_processes} processes"
    )
    print(f"  Model: {model_path}")
    print(f"  Equilibrium ENV: lam={ENV['lam']}, sim_time={ENV['sim_time']}, r={ENV['r']}.\n")

    with mp.Pool(n_processes) as pool:
        for bg_idx, mlp_mean in tqdm(
            pool.imap_unordered(_run_mlp_cell, tasks),
            total=n_bg,
            desc="MLP cells completed",
        ):
            mlp_column[bg_idx] = mlp_mean

    return mlp_column


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_extended_results(zi_df: pd.DataFrame, mlp_column: np.ndarray,
                             model_label: str = MLP_LABEL):
    """Display 10×11 advantage table and best-deviation analysis."""
    n = len(zi_df)

    # Build extended DataFrame
    mlp_series = pd.Series(mlp_column, index=zi_df.index, name=model_label)
    df = pd.concat([zi_df, mlp_series], axis=1)

    # ── Advantage matrix ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DEVIATOR PROFIT TABLE  (extended with MLP-SAC deviator)")
    print("Rows = background strategy  |  Cols = deviator strategy")
    print(f"Value = mean deviator profit  |  Last col ({model_label}) = SAC-trained MLP")
    print("=" * 90)
    print(df.round(1).to_string())

    # ── Best deviator per row ─────────────────────────────────────────────────
    best_dev = df.idxmax(axis=1)
    best_adv = df.max(axis=1)

    print("\n" + "=" * 90)
    print("BEST DEVIATOR STRATEGY FOR EACH BACKGROUND STRATEGY")
    print("=" * 90)
    print(f"{'BG Strategy':>12} │ {'Best Deviation':>14} │ {'Advantage':>10} │ NE? │ MLP better than ZI best?")
    print("─" * 70)

    ne_candidates   = []
    mlp_beats_zi    = []

    for bg_label in df.index:
        best     = best_dev[bg_label]
        adv      = best_adv[bg_label]
        zi_best  = zi_df.idxmax(axis=1)[bg_label]
        zi_adv   = zi_df.max(axis=1)[bg_label]
        mlp_val  = mlp_column[list(df.index).index(bg_label)]
        is_ne    = (bg_label == best)
        mlp_wins = (best == model_label)

        ne_str  = " ✓" if is_ne else ""
        win_str = f"Yes (+{mlp_val - zi_adv:+.0f})" if mlp_wins else f"No  ({zi_best} +{zi_adv:.0f})"

        if is_ne:
            ne_candidates.append(bg_label)
        if mlp_wins:
            mlp_beats_zi.append(bg_label)

        print(f"{bg_label:>12} │ {best:>14} │ {adv:>10.1f}{ne_str:3s} │     │ {win_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  MLP is the best deviator against {len(mlp_beats_zi)}/10 background strategies:")
    for bg in mlp_beats_zi:
        bg_idx  = list(df.index).index(bg)
        s       = STRATEGIES[int(bg[1:])]
        zi_best = zi_df.idxmax(axis=1)[bg]
        zi_adv  = zi_df.max(axis=1)[bg]
        print(f"    {bg} (bg={s['shade']} η={s['eta']})  MLP={mlp_column[bg_idx]:.1f}  "
              f"best-ZI={zi_best} ({zi_adv:.1f})  Δ={mlp_column[bg_idx]-zi_adv:+.1f}")

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
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Path to SB3 SAC model zip (default: {DEFAULT_MODEL})")
    p.add_argument("--num-runs", type=int, default=DEFAULT_RUNS,
                   help=f"Simulations per MLP-deviator cell (default: {DEFAULT_RUNS})")
    p.add_argument("--processes", type=int, default=None,
                   help="Worker processes (default: cpu_count)")
    p.add_argument("--output", default="results/equilibrium/equilibrium_mlp_results.csv",
                   help="Output CSV path (default: results/equilibrium/equilibrium_mlp_results.csv)")
    p.add_argument("--env", type=str, default="B", choices=["A", "B", "C"],
                   help="Market environment preset (A/B/C). Sets lam, shock_var, pv_var. (default: B)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Apply environment preset ─────────────────────────────────────────────
    preset = _ENV_PRESETS[args.env]
    ENV["lam"]       = preset["lam"]
    ENV["shock_var"] = preset["shock_var"]
    ENV["pv_var"]    = preset["pv_var"]
    print(f"Environment: {args.env}  |  lam={preset['lam']}  shock_var={preset['shock_var']}  pv_var={preset['pv_var']}")

    # ── Load ZI×ZI baseline results ──────────────────────────────────────────
    print(f"Loading ZI×ZI baseline from {BASELINE_CSV} ...")
    zi_df = pd.read_csv(BASELINE_CSV, index_col=0)
    print(f"  Loaded {zi_df.shape[0]}×{zi_df.shape[1]} matrix.\n")

    # ── Run MLP deviator column ───────────────────────────────────────────────
    mlp_column = run_mlp_column(
        model_path=args.model,
        num_runs=args.num_runs,
        n_processes=args.processes,
    )

    # ── Display extended results ──────────────────────────────────────────────
    ext_df = display_extended_results(zi_df, mlp_column)

    # ── Save ─────────────────────────────────────────────────────────────────
    ext_df.to_csv(args.output)
    print(f"\nExtended advantage matrix saved to {args.output}")


if __name__ == "__main__":
    main()

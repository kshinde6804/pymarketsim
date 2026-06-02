"""
tron2_env.py — Gymnasium environment for TRON2 training (ICAIF24, §4.3, §5.3).

Mirrors ZIEnv exactly except:
  1. Action space is MultiDiscrete([21, 21]) → decoded via SHADE_BINS / ETA_BINS.
  2. Background agents are configured by ne_strategy index (0–9) from STRATEGIES.
  3. Env parameters (lam, shock_var, pv_var, normalizers) come from ENV_CONFIGS.

Observation space: Box(14,) — identical to ZIEnv:
    [0]  time_left       [1]  fundamental    [2]  best_ask      [3]  best_bid
    [4]  inventory       [5]  midprice_delta [6]  vol_imbalance [7]  queue_imbalance
    [8]  realized_vol    [9]  rsi            [10] est_fundamental
    [11] pv_buy          [12] pv_sell        [13] side  (0=BUY, 1=SELL)

Action space: MultiDiscrete([21, 21]) → [shade_idx, eta_idx]
    shade = SHADE_BINS[shade_idx]   eta = ETA_BINS[eta_idx]
"""

import heapq
import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as dist
from gymnasium import spaces

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.wrappers.metrics import (
    midprice_move,
    queue_imbalance,
    realized_volatility,
    relative_strength_index,
    volume_imbalance,
)

# ── Bin constants (change SHADE_MAX here to adjust discretisation) ─────────────
SHADE_MAX  = 1000
SHADE_BINS = np.linspace(0, SHADE_MAX, 21)   # (21,) shade values
ETA_BINS   = np.linspace(0, 1, 21)           # (21,) eta values

# ── Per-environment parameters (Table 3, §5.2) ────────────────────────────────
ENV_CONFIGS = {
    "A": {
        "lam":       0.0005,
        "shock_var": 1e6,
        "pv_var":    5e6,
        "normalizers": {"fundamental": 1e5, "invt": 10, "reward": 1e2, "pv": 5e5},
    },
    "B": {
        "lam":       0.005,
        "shock_var": 1e6,
        "pv_var":    5e6,
        "normalizers": {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    },
    "C": {
        "lam":       0.012,
        "shock_var": 2e4,
        "pv_var":    2e7,
        "normalizers": {"fundamental": 1e5, "invt": 10, "reward": 1e4, "pv": 5e5},
    },
}

# ── Background ZI strategies (§4.1, S0–S9) ────────────────────────────────────
STRATEGIES = {
    0: {"shade": [0,    450],  "eta": 0.5},
    1: {"shade": [0,    600],  "eta": 0.5},
    2: {"shade": [90,   110],  "eta": 0.5},
    3: {"shade": [140,  160],  "eta": 0.5},
    4: {"shade": [190,  210],  "eta": 0.5},
    5: {"shade": [280,  320],  "eta": 0.5},
    6: {"shade": [380,  420],  "eta": 0.5},
    7: {"shade": [380,  420],  "eta": 1.0},
    8: {"shade": [460,  540],  "eta": 0.5},   # default equilibrium strategy
    9: {"shade": [950, 1050],  "eta": 0.5},
}

# Shared simulation parameters
_MEAN     = 1e5
_R        = 0.01
_Q_MAX    = 10
_SIM_TIME = 2000
_N_BG     = 24   # 24 background + 1 TRON = 25 total (paper §5.2)
_ARRIVALS_SAMPLED = 10000


def _sample_arrivals(p: float, n: int):
    """Sample n inter-arrival gaps from Geometric(p)."""
    return dist.Geometric(torch.tensor([p])).sample((n,)).squeeze()


class TRONEnv2(gym.Env):
    """Gymnasium wrapper for TRON2 training.

    Parameters
    ----------
    env_tag : str
        One of 'A', 'B', 'C' — selects market parameters from ENV_CONFIGS.
    ne_strategy : int
        Index 0–9 into STRATEGIES; configures all 24 background ZI agents.
        Default: 8 (S8, equilibrium strategy for Env C).
    """

    metadata = {"render_modes": []}

    def __init__(self, env_tag: str = "B", ne_strategy: int = 8):
        super().__init__()

        if env_tag not in ENV_CONFIGS:
            raise ValueError(f"env_tag must be one of {list(ENV_CONFIGS)}; got {env_tag!r}")
        if ne_strategy not in STRATEGIES:
            raise ValueError(f"ne_strategy must be 0–9; got {ne_strategy}")

        cfg = ENV_CONFIGS[env_tag]
        strat = STRATEGIES[ne_strategy]

        self.env_tag     = env_tag
        self.ne_strategy = ne_strategy
        self.lam         = cfg["lam"]
        self.shock_var   = cfg["shock_var"]
        self.pv_var      = cfg["pv_var"]
        self.normalizers = dict(cfg["normalizers"])
        self.bg_shade    = strat["shade"]
        self.bg_eta      = strat["eta"]

        self.mean     = _MEAN
        self.r        = _R
        self.q_max    = _Q_MAX
        self.sim_time = _SIM_TIME
        self.n_bg     = _N_BG

        self.time             = 0
        self.last_value       = 0.0
        self.final_fundamental = 0.0
        self.current_side      = BUY

        # ── Observation / action spaces ────────────────────────────────────────
        lower = np.array([0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0], dtype=np.float64)
        upper = np.array([1, 1, 1, 1,  1,  1,  1,  1, 1, 1, 1,  1,  1, 1], dtype=np.float64)
        self.observation_space = spaces.Box(low=lower, high=upper, shape=(14,), dtype=np.float64)
        self.action_space      = spaces.MultiDiscrete([21, 21])
        self.observation       = np.zeros(14, dtype=np.float64)

        # ── Market (rebuilt each reset) ────────────────────────────────────────
        fundamental = LazyGaussianMeanReverting(
            mean=self.mean, final_time=self.sim_time + 1, r=self.r, shock_var=self.shock_var
        )
        self.market = Market(fundamental=fundamental, time_steps=self.sim_time)

        # ── Background agents ──────────────────────────────────────────────────
        self.agents: dict = {}
        for agent_id in range(self.n_bg):
            self.agents[agent_id] = ZIAgent(
                agent_id=agent_id,
                market=self.market,
                q_max=self.q_max,
                shade=self.bg_shade,
                pv_var=self.pv_var,
                eta=self.bg_eta,
            )

        # ── RL agent ───────────────────────────────────────────────────────────
        self.rl_agent_id = self.n_bg
        self.rl_agent    = ZIAgent(
            agent_id=self.rl_agent_id,
            market=self.market,
            q_max=self.q_max,
            shade=[0, 0],     # placeholder; orders placed directly by step()
            pv_var=self.pv_var,
        )

        # ── Arrival buffers ────────────────────────────────────────────────────
        # BG events stored as min-heap of (time, agent_id); RL tracked separately.
        self.event_heap: list  = []
        self.next_rl_time: int = 0
        self.arrival_times_bg  = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
        self.arrival_index_bg  = 0
        self.arrival_times_rl  = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
        self.arrival_index_rl  = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time       = 0
        self.last_value = 0.0
        self.observation = np.zeros(14, dtype=np.float64)

        # New fundamental; prefetch so F_T is available immediately.
        fundamental = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.sim_time + 1,
            r=self.r,
            shock_var=self.shock_var,
        )
        fundamental.prefetch_all()
        self.final_fundamental = fundamental.get_final_fundamental()
        self.market.reset(fundamental=fundamental)

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()
        self.rl_agent.reset()

        self._reset_arrivals()

        self._run_until_next_rl_arrival()

        return self.get_obs(), {}

    def step(self, action):
        if self.time >= self.sim_time:
            return self._end_sim()

        shade_idx, eta_idx = int(action[0]), int(action[1])
        shade = float(SHADE_BINS[shade_idx])
        eta   = float(ETA_BINS[eta_idx])

        self._rl_agent_step(shade, eta)
        self._agents_step()
        reward = self._market_step(agent_only=False)
        self.time += 1

        ended = self._run_until_next_rl_arrival()
        if ended:
            obs, end_r, term, trunc, info = self._end_sim()
            return obs, reward + end_r, term, trunc, info

        return self.get_obs(), reward, False, False, {}

    def render(self):
        pass

    def get_episode_profit(self) -> float:
        """Raw (unnormalised) profit for the RL agent at current state."""
        return (
            self.rl_agent.position * self.final_fundamental
            + self.rl_agent.cash
            + self.rl_agent.get_pos_value()
        )

    # ── Arrival helpers ────────────────────────────────────────────────────────

    def _reset_arrivals(self):
        self.event_heap = []
        self.arrival_times_bg = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
        self.arrival_index_bg = 0

        # Ensure first RL arrival falls within sim_time (may be rare but possible).
        while True:
            self.arrival_times_rl = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
            if self.arrival_times_rl[0].item() < self.sim_time:
                break
        self.arrival_index_rl = 0

        # Schedule initial BG arrivals into heap
        for agent_id in range(self.n_bg):
            t = int(self.arrival_times_bg[self.arrival_index_bg].item())
            heapq.heappush(self.event_heap, (t, agent_id))
            self.arrival_index_bg += 1

        # Track first RL arrival as a scalar (not in the heap)
        self.next_rl_time = int(self.arrival_times_rl[self.arrival_index_rl].item())
        self.arrival_index_rl += 1

    def _schedule_next_bg(self, agent_id: int):
        """Push agent_id's next BG arrival into the heap."""
        if self.arrival_index_bg == _ARRIVALS_SAMPLED:
            self.arrival_times_bg = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
            self.arrival_index_bg = 0
        gap = int(self.arrival_times_bg[self.arrival_index_bg].item())
        heapq.heappush(self.event_heap, (self.time + 1 + gap, agent_id))
        self.arrival_index_bg += 1

    def _schedule_next_rl(self):
        """Advance next_rl_time by one inter-arrival gap."""
        if self.arrival_index_rl == _ARRIVALS_SAMPLED:
            self.arrival_times_rl = _sample_arrivals(self.lam, _ARRIVALS_SAMPLED)
            self.arrival_index_rl = 0
        gap = int(self.arrival_times_rl[self.arrival_index_rl].item())
        self.next_rl_time = self.time + 1 + gap
        self.arrival_index_rl += 1

    def _run_until_next_rl_arrival(self) -> bool:
        """Jump directly to each BG event, then stop at the next RL arrival.

        Returns True if the episode ended (next_rl_time >= sim_time).
        On RL arrival: pre-assigns current_side, calls _update_obs(), returns False.
        BG agents that coincide with the RL arrival stay in event_heap; _agents_step()
        in step() will drain them at self.time after RL acts.
        """
        while True:
            if self.next_rl_time >= self.sim_time:
                self.time = self.sim_time
                return True

            next_bg_t = self.event_heap[0][0] if self.event_heap else self.sim_time
            if next_bg_t >= self.next_rl_time:
                # No more BG events before (or at same time as) RL — jump to RL.
                self.time = self.next_rl_time
                self.current_side = random.choice([BUY, SELL])
                self._update_obs()
                return False

            # BG event arrives before RL: process it and continue.
            self.time = next_bg_t
            self.market.event_queue.set_time(self.time)
            while self.event_heap and self.event_heap[0][0] == next_bg_t:
                _, agent_id = heapq.heappop(self.event_heap)
                agent = self.agents[agent_id]
                self.market.withdraw_all(agent_id)
                self.market.add_orders(agent.take_action())
                self._schedule_next_bg(agent_id)
            self._market_step(agent_only=True)

    # ── Per-step helpers ───────────────────────────────────────────────────────

    def _rl_agent_step(self, shade: float, eta: float):
        """Submit a ZI-style limit order for the RL agent using decoded shade/eta."""
        self.market.event_queue.set_time(self.time)
        self.market.withdraw_all(self.rl_agent_id)

        t        = self.market.get_time()
        side     = self.current_side
        estimate = self.rl_agent.estimate_fundamental()
        pv_value = self.rl_agent.pv.value_for_exchange(self.rl_agent.position, side)

        if side == BUY:
            price = estimate + pv_value - shade
        else:
            price = estimate + pv_value + shade

        if eta < 1.0:
            base_price = estimate + pv_value
            if side == BUY:
                best = self.market.order_book.get_best_ask()
                if not math.isinf(best) and (base_price - best) > eta * shade:
                    price = best
            else:
                best = self.market.order_book.get_best_bid()
                if not math.isinf(best) and (best - base_price) > eta * shade:
                    price = best

        self.rl_agent._order_counter += 1
        order_id = self.rl_agent_id * 1000000 + self.rl_agent._order_counter
        self.market.add_orders(
            [Order(price=price, quantity=1, agent_id=self.rl_agent_id,
                   time=t, order_type=side, order_id=order_id)]
        )

        self._schedule_next_rl()

    def _agents_step(self):
        """Process BG agents that coincide with the RL arrival at self.time."""
        if not self.event_heap or self.event_heap[0][0] != self.time:
            return
        self.market.event_queue.set_time(self.time)
        while self.event_heap and self.event_heap[0][0] == self.time:
            _, agent_id = heapq.heappop(self.event_heap)
            agent = self.agents[agent_id]
            self.market.withdraw_all(agent_id)
            self.market.add_orders(agent.take_action())
            self._schedule_next_bg(agent_id)

    def _market_step(self, agent_only: bool = True) -> float:
        """Clear market, update positions, optionally compute reward.

        Always syncs event_queue to self.time before market.step() to prevent
        dt<0 in LazyGaussianMeanReverting._generate_at().
        """
        self.market.event_queue.set_time(self.time)
        new_orders = self.market.step()

        for matched in new_orders:
            aid  = matched.order.agent_id
            qty  = matched.order.order_type * matched.order.quantity
            cash = -matched.price * matched.order.quantity * matched.order.order_type
            if aid == self.rl_agent_id:
                self.rl_agent.update_position(qty, cash)
            elif aid in self.agents:
                self.agents[aid].update_position(qty, cash)

        if not agent_only:
            current_value = (
                self.rl_agent.position * self.final_fundamental
                + self.rl_agent.cash
                + self.rl_agent.get_pos_value()
            )
            reward = (current_value - self.last_value) / self.normalizers["reward"]
            self.last_value = current_value
            return reward

        return 0.0

    def _end_sim(self):
        current_value = (
            self.rl_agent.position * self.final_fundamental
            + self.rl_agent.cash
            + self.rl_agent.get_pos_value()
        )
        reward = (current_value - self.last_value) / self.normalizers["reward"]
        return self.get_obs(), reward, True, False, {}

    # ── Observation ────────────────────────────────────────────────────────────

    def _env_estimate_fundamental(self) -> float:
        """Bayesian fundamental estimate using env clock (self.time), not market clock."""
        mean, r, T = self.market.get_info()
        val  = self.market.fundamental.get_value_at(self.time)
        rho  = (1.0 - r) ** (T - self.time)
        return (1.0 - rho) * mean + rho * val

    def _update_obs(self):
        t                = self.time
        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask         = self.market.order_book.get_best_ask()
        best_bid         = self.market.order_book.get_best_bid()

        mp_delta  = midprice_move(self.market)
        vol_imb   = volume_imbalance(self.market)
        que_imb   = queue_imbalance(self.market)
        rv        = realized_volatility(self.market)
        rsi       = relative_strength_index(self.market)
        est_fund  = self._env_estimate_fundamental()
        pv_buy    = self.rl_agent.pv.value_for_exchange(self.rl_agent.position, BUY)
        pv_sell   = self.rl_agent.pv.value_for_exchange(self.rl_agent.position, SELL)

        self.observation = self._normalize(
            t, fundamental_value, best_ask, best_bid,
            self.rl_agent.position, mp_delta, vol_imb, que_imb,
            rv, rsi, est_fund, pv_buy, pv_sell, self.current_side,
        )

    def _normalize(
        self, t, fundamental_value, best_ask, best_bid, inventory,
        mp_delta, vol_imb, que_imb, rv, rsi, est_fund,
        pv_buy, pv_sell, side,
    ) -> np.ndarray:
        N     = self.normalizers
        pv_n  = N["pv"]

        time_left_n = (self.sim_time - t) / self.sim_time
        fund_n      = fundamental_value / N["fundamental"]
        ask_n       = 1.0 if math.isinf(best_ask) else best_ask / N["fundamental"]
        bid_n       = 0.0 if math.isinf(best_bid) else best_bid / N["fundamental"]
        inv_n       = float(np.clip(inventory / N["invt"], -1.0, 1.0))
        mp_n        = float(np.clip(mp_delta / 1e2, -1.0, 1.0))
        rsi_n       = rsi / 100.0
        est_n       = est_fund / N["fundamental"]
        pv_buy_n    = float(np.clip(pv_buy  / pv_n, -1.0, 1.0))
        pv_sell_n   = float(np.clip(pv_sell / pv_n, -1.0, 1.0))
        side_n      = 0.0 if side == BUY else 1.0

        obs = np.array(
            [
                time_left_n, fund_n, ask_n, bid_n, inv_n,
                mp_n, vol_imb, que_imb,
                float(np.clip(rv, 0.0, 1.0)),
                rsi_n, est_n, pv_buy_n, pv_sell_n, side_n,
            ],
            dtype=np.float64,
        )
        if np.isnan(obs[9]):   # RSI → neutral when no history
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_obs(self) -> np.ndarray:
        return self.observation

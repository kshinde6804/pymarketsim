"""
ZIEnv — Gymnasium environment for training a ZI-style RL agent.

The RL agent controls `shade` (surplus demanded) and `eta` (liquidity-taking
threshold) via a 2-dimensional continuous action space.  Background ZI traders
provide market liquidity according to independent Poisson arrival processes.

Observation  (13 features, all in [-1, 1] or [0, 1]):
    [0]  time_left          (sim_time - t) / sim_time
    [1]  fundamental        observed fundamental / N["fundamental"]
    [2]  best_ask           best ask / N["fundamental"];  1.0 when book empty
    [3]  best_bid           best bid / N["fundamental"];  0.0 when book empty
    [4]  inventory          clip(position / N["invt"], -1, 1)
    [5]  midprice_delta     clip(midprice_move / 1e2,  -1, 1)
    [6]  vol_imbalance      volume_imbalance(market)   ∈ [-1, 1]
    [7]  queue_imbalance    queue_imbalance(market)    ∈ [-1, 1]
    [8]  realized_vol       clip(realized_volatility,   0, 1)
    [9]  rsi                relative_strength_index / 100
    [10] est_fundamental    Bayesian fundamental estimate / N["fundamental"]
    [11] pv_buy             clip(pv.value_for_exchange(pos, BUY)  / N["pv"], -1, 1)
    [12] pv_sell            clip(pv.value_for_exchange(pos, SELL) / N["pv"], -1, 1)

Action  (2 values in [0, 1]):
    [0]  shade_norm  →  shade = shade_norm * (shade_max - shade_min) + shade_min
    [1]  eta         ∈ [0, 1] directly
"""

import math
import random
from collections import defaultdict

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


def sample_arrivals(p, num_samples):
    """Sample inter-arrival gaps from a Geometric(p) distribution."""
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()


class ZIEnv(gym.Env):
    """Gymnasium environment for training a ZI-style RL agent in PyMarketSim."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_background_agents: int,
        sim_time: int,
        lam: float = 0.1,
        lam_zi: float = 0.1,
        mean: float = 1e5,
        r: float = 0.05,
        shock_var: float = 5e6,
        q_max: int = 10,
        pv_var: float = 5e6,
        shade=None,
        shade_range=None,
        normalizers=None,
        bg_strategies=None,
        warmup_fraction: float = 0.1,
    ):
        """
        Args:
            num_background_agents: Number of background ZI traders.
            sim_time:              Total simulation time steps per episode.
            lam:                   Arrival rate for background agents (Poisson).
            lam_zi:                Arrival rate for the RL agent (Poisson).
            mean:                  Long-run fundamental mean.
            r:                     Mean-reversion rate.
            shock_var:             Variance of fundamental shocks.
            q_max:                 Maximum absolute inventory position.
            pv_var:                Variance of private value draws.
            shade:                 Background agents' shade range [low, high].
            shade_range:           RL agent shade denormalization [d_min, d_max].
                                   action[0] * (d_max - d_min) + d_min = shade.
            normalizers:           Dict with keys "fundamental", "invt",
                                   "reward", "pv" for observation/reward scaling.
            bg_strategies:         Optional list of dicts, each with keys
                                   'shade' ([lo, hi]) and 'eta' (float).
                                   If provided, each reset() randomly picks one
                                   strategy for all background agents, exposing
                                   the RL agent to diverse market conditions.
                                   Example: [{'shade': [0,600], 'eta': 0.5},
                                             {'shade': [380,420], 'eta': 1.0}]
            warmup_fraction:       Fraction of sim_time to run background agents
                                   before the RL agent acts (populates order book).
                                   Set to 0.0 for sparse markets (lam << 0.1) to
                                   avoid the RL agent's first arrival being pushed
                                   beyond sim_time by repeated warm-up reschedules.
                                   Default: 0.1.
        """
        super().__init__()

        if shade is None:
            shade = [250, 500]
        if shade_range is None:
            shade_range = [10, 500]
        if normalizers is None:
            normalizers = {"fundamental": 1e5, "invt": 10, "reward": 1e4, "pv": 5e5}

        self.num_agents = num_background_agents
        self.sim_time = sim_time
        self.lam = lam
        self.lam_zi = lam_zi
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade
        self.shade_range = shade_range
        self.normalizers = normalizers
        self.bg_strategies = bg_strategies
        self.warmup_fraction = warmup_fraction
        self.time = 0
        self.last_value = 0.0

        # ── Arrival buffers ──────────────────────────────────────────────
        self.arrivals_sampled = 10000

        self.arrivals = defaultdict(list)
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_zi = defaultdict(list)
        self.arrival_times_zi = sample_arrivals(lam_zi, self.arrivals_sampled)
        self.arrival_index_zi = 0

        # ── Market ───────────────────────────────────────────────────────
        fundamental = LazyGaussianMeanReverting(
            mean=mean, final_time=sim_time + 1, r=r, shock_var=shock_var
        )
        self.market = Market(fundamental=fundamental, time_steps=sim_time)

        # ── Background agents ─────────────────────────────────────────────
        self.agents = {}
        for agent_id in range(num_background_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(
                agent_id
            )
            self.arrival_index += 1
            self.agents[agent_id] = ZIAgent(
                agent_id=agent_id,
                market=self.market,
                q_max=q_max,
                shade=shade,
                pv_var=pv_var,
            )

        # ── RL agent ──────────────────────────────────────────────────────
        self.zi_agent_id = num_background_agents
        # Guarantee first RL arrival is within sim_time.
        first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        while first_zi >= sim_time:
            self.arrival_times_zi = sample_arrivals(lam_zi, self.arrivals_sampled)
            self.arrival_index_zi = 0
            first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        self.arrivals_zi[first_zi].append(self.zi_agent_id)
        self.arrival_index_zi += 1
        self.zi_agent = ZIAgent(
            agent_id=self.zi_agent_id,
            market=self.market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
        )

        # ── Gym spaces ────────────────────────────────────────────────────
        lower_bound = np.array(
            [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1], dtype=np.float64
        )
        upper_bound = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=lower_bound, high=upper_bound, shape=(13,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation = np.zeros(13, dtype=np.float64)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.last_value = 0.0
        self.observation = np.zeros(13, dtype=np.float64)

        # New fundamental for each episode
        fundamental = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.sim_time + 1,
            r=self.r,
            shock_var=self.shock_var,
        )
        self.market.reset(fundamental=fundamental)

        # If multi-strategy mode, pick a random background strategy for this episode
        if self.bg_strategies is not None:
            chosen = random.choice(self.bg_strategies)
            for agent_id in self.agents:
                self.agents[agent_id].shade = chosen['shade']
                self.agents[agent_id].eta   = chosen.get('eta', 1.0)

        # Reset all agents
        for agent_id in self.agents:
            self.agents[agent_id].reset()
        self.zi_agent.reset()

        # Resample arrival schedules
        self.reset_arrivals()

        # Warm-up: run background agents to populate the order book
        self.run_agents_only()

        end = self.run_until_next_zi_arrival()
        if end:
            raise ValueError(
                "ZIEnv: episode ended before RL agent arrived. "
                "Increase sim_time or lam_zi."
            )

        return self.get_obs(), {}

    def step(self, action):
        if self.time >= self.sim_time:
            return self.end_sim()

        self.zi_agent_step(action)
        self.agents_step()
        reward = self.market_step(agent_only=False)
        self.time += 1

        end = self.run_until_next_zi_arrival()
        if end:
            return self.end_sim()

        return self.get_obs(), reward, False, False, {}

    def render(self):
        pass

    # ── Arrival helpers ────────────────────────────────────────────────────

    def reset_arrivals(self):
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_zi = defaultdict(list)
        self.arrival_times_zi = sample_arrivals(self.lam_zi, self.arrivals_sampled)
        self.arrival_index_zi = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(
                agent_id
            )
            self.arrival_index += 1

        # Guarantee first RL arrival is within sim_time.
        first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        while first_zi >= self.sim_time:
            self.arrival_times_zi = sample_arrivals(self.lam_zi, self.arrivals_sampled)
            self.arrival_index_zi = 0
            first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        self.arrivals_zi[first_zi].append(self.zi_agent_id)
        self.arrival_index_zi += 1

    def run_agents_only(self):
        """Warm-up: advance background agents for warmup_fraction of sim_time.

        Any RL agent arrivals that fall inside the warm-up window are
        skipped but immediately rescheduled for after the current time,
        ensuring run_until_next_zi_arrival() always finds a future arrival.

        Set warmup_fraction=0.0 for sparse markets (e.g. lam=0.005) to avoid
        repeated reschedules pushing the RL agent's first arrival beyond sim_time.
        """
        for _ in range(int(self.warmup_fraction * self.sim_time)):
            if self.arrivals[self.time]:
                self.agents_step()
                self.market_step(agent_only=True)
            # If RL agent was scheduled during warm-up, reschedule it forward
            if self.arrivals_zi[self.time]:
                if self.arrival_index_zi == self.arrivals_sampled:
                    self.arrival_times_zi = sample_arrivals(
                        self.lam_zi, self.arrivals_sampled
                    )
                    self.arrival_index_zi = 0
                self.arrivals_zi[
                    self.arrival_times_zi[self.arrival_index_zi].item() + 1 + self.time
                ].append(self.zi_agent_id)
                self.arrival_index_zi += 1
            self.time += 1

    def run_until_next_zi_arrival(self):
        """Advance market until the RL agent's next scheduled arrival.

        Returns:
            True  if the episode ended (time >= sim_time) before RL agent arrived.
            False if RL agent arrived; also calls update_obs().
        """
        while len(self.arrivals_zi[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            self.market_step(agent_only=True)
            self.time += 1

        if self.time >= self.sim_time:
            return True
        else:
            self.update_obs()
            return False

    # ── Per-step helpers ───────────────────────────────────────────────────

    def zi_agent_step(self, action):
        """Translate gym action [shade_norm, eta] → limit order, submit to market."""
        d_min, d_max = self.shade_range
        shade_val = float(action[0]) * (d_max - d_min) + d_min
        eta = float(action[1])

        self.market.event_queue.set_time(self.time)
        self.market.withdraw_all(self.zi_agent_id)

        t = self.market.get_time()
        side = random.choice([BUY, SELL])
        estimate = self.zi_agent.estimate_fundamental()
        pv_value = self.zi_agent.pv.value_for_exchange(self.zi_agent.position, side)

        if side == BUY:
            price = estimate + pv_value - shade_val
        else:
            price = estimate + pv_value + shade_val

        # eta: take liquidity if surplus fraction exceeds threshold
        # Mirrors zero_intelligence_agent.py:54-63 exactly.
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

        self.zi_agent._order_counter += 1
        order_id = self.zi_agent_id * 1000000 + self.zi_agent._order_counter
        self.market.add_orders(
            [
                Order(
                    price=price,
                    quantity=1,
                    agent_id=self.zi_agent_id,
                    time=t,
                    order_type=side,
                    order_id=order_id,
                )
            ]
        )

        # Schedule next RL agent arrival
        if self.arrival_index_zi == self.arrivals_sampled:
            self.arrival_times_zi = sample_arrivals(
                self.lam_zi, self.arrivals_sampled
            )
            self.arrival_index_zi = 0
        self.arrivals_zi[
            self.arrival_times_zi[self.arrival_index_zi].item() + 1 + self.time
        ].append(self.zi_agent_id)
        self.arrival_index_zi += 1

    def agents_step(self):
        """Let all background agents that arrive at self.time act."""
        agents = self.arrivals[self.time]
        if len(agents) == 0:
            return
        self.market.event_queue.set_time(self.time)
        for agent_id in agents:
            agent = self.agents[agent_id]
            self.market.withdraw_all(agent_id)
            orders = agent.take_action()
            self.market.add_orders(orders)

            if self.arrival_index == self.arrivals_sampled:
                self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                self.arrival_index = 0
            self.arrivals[
                self.arrival_times[self.arrival_index].item() + 1 + self.time
            ].append(agent_id)
            self.arrival_index += 1

    def market_step(self, agent_only=True):
        """Clear the market, update positions, and optionally compute reward.

        Always syncs event_queue to self.time before market.step() so that
        market.get_time() stays consistent with the env clock even when
        agents_step() was a no-op (no arrivals → no set_time() call there).
        This prevents dt<0 in LazyGaussianMeanReverting._generate_at() which
        requires strictly increasing time access.

        Args:
            agent_only: If False, compute and return the interim reward.

        Returns:
            Reward (float) when agent_only=False, else 0.0.
        """
        self.market.event_queue.set_time(self.time)
        new_orders = self.market.step()
        for matched_order in new_orders:
            aid = matched_order.order.agent_id
            qty = matched_order.order.order_type * matched_order.order.quantity
            cash = (
                -matched_order.price
                * matched_order.order.quantity
                * matched_order.order.order_type
            )
            if aid == self.zi_agent_id:
                self.zi_agent.update_position(qty, cash)
            else:
                self.agents[aid].update_position(qty, cash)

        if not agent_only:
            # Mark-to-market using the Bayesian estimated fundamental.
            # Computed using self.time (env clock) so the fundamental lookup
            # stays strictly forward even after market.step() increments the
            # internal event_queue clock to self.time+1.
            est_fund = self._env_estimate_fundamental()
            current_value = (
                self.zi_agent.position * est_fund
                + self.zi_agent.cash
                + self.zi_agent.get_pos_value()
            )
            reward = (current_value - self.last_value) / self.normalizers["reward"]
            self.last_value = current_value
            return reward

        return 0.0

    def end_sim(self):
        final_fund = self.market.get_final_fundamental()
        current_value = (
            self.zi_agent.position * final_fund
            + self.zi_agent.cash
            + self.zi_agent.get_pos_value()
        )
        reward = (current_value - self.last_value) / self.normalizers["reward"]
        return self.get_obs(), reward, True, False, {}

    # ── Observation ────────────────────────────────────────────────────────

    def _env_estimate_fundamental(self) -> float:
        """Bayesian fundamental estimate using self.time (env clock).

        Bypasses market.get_time() so the fundamental is always accessed in
        forward order regardless of the event_queue's internal clock state.
        """
        mean, r, T = self.market.get_info()
        t = self.time
        val = self.market.fundamental.get_value_at(t)
        rho = (1 - r) ** (T - t)
        return (1 - rho) * mean + rho * val

    def update_obs(self):
        t = self.time
        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask = self.market.order_book.get_best_ask()
        best_bid = self.market.order_book.get_best_bid()
        inventory = self.zi_agent.position

        mp_delta = midprice_move(self.market)
        vol_imb = volume_imbalance(self.market)
        que_imb = queue_imbalance(self.market)
        rv = realized_volatility(self.market)
        rsi = relative_strength_index(self.market)

        est_fund = self._env_estimate_fundamental()
        pv_buy = self.zi_agent.pv.value_for_exchange(self.zi_agent.position, BUY)
        pv_sell = self.zi_agent.pv.value_for_exchange(self.zi_agent.position, SELL)

        self.observation = self.normalize(
            t,
            fundamental_value,
            best_ask,
            best_bid,
            inventory,
            mp_delta,
            vol_imb,
            que_imb,
            rv,
            rsi,
            est_fund,
            pv_buy,
            pv_sell,
        )

    def normalize(
        self,
        t,
        fundamental_value,
        best_ask,
        best_bid,
        inventory,
        mp_delta,
        vol_imb,
        que_imb,
        rv,
        rsi,
        est_fund,
        pv_buy,
        pv_sell,
    ):
        N = self.normalizers
        pv_n = N.get("pv", 5e5)

        time_left_n = (self.sim_time - t) / self.sim_time
        fund_n = fundamental_value / N["fundamental"]
        ask_n = 1.0 if math.isinf(best_ask) else best_ask / N["fundamental"]
        bid_n = 0.0 if math.isinf(best_bid) else best_bid / N["fundamental"]
        inv_n = np.clip(inventory / N["invt"], -1.0, 1.0)
        mp_n = np.clip(mp_delta / 1e2, -1.0, 1.0)
        rsi_n = rsi / 100.0
        est_n = est_fund / N["fundamental"]
        pv_buy_n = np.clip(pv_buy / pv_n, -1.0, 1.0)
        pv_sell_n = np.clip(pv_sell / pv_n, -1.0, 1.0)

        obs = np.array(
            [
                time_left_n,
                fund_n,
                ask_n,
                bid_n,
                inv_n,
                mp_n,
                vol_imb,
                que_imb,
                np.clip(rv, 0.0, 1.0),
                rsi_n,
                est_n,
                pv_buy_n,
                pv_sell_n,
            ],
            dtype=np.float64,
        )
        # Guard against NaN/inf from metrics when the order book has little history
        # (sparse arrivals or early in an episode before the warm-up populates the book).
        # RSI defaults to 0.5 (neutral = 50/100); all other NaN features default to 0.
        if np.isnan(obs[9]):   # rsi_n
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_obs(self):
        return self.observation

"""
TRONEnv — Gymnasium environment for training the TRON recurrent DQN agent.

Mirrors ZIEnv with the following differences:
  - Observation space: Box(14,) — 13 ZIEnv features + side indicator [13]
  - Action space:      MultiDiscrete([n_shade, n_eta]) — derived from TRONPolicy bins
  - Side pre-assignment: env samples BUY/SELL before the agent acts and
    includes it in the observation so the agent can condition on it.

The RL agent and all background agents share a single arrival pool — no
explicit warm-up phase.  Both use Geometric inter-arrival sampling;
background agents use rate `lam`, the RL agent uses rate `lam_zi`.

Discrete bins (must match TRONPolicy):
  shade_bins = np.linspace(0, 600, 21)   # 21 uniform bins (paper §4.3)
  eta_bins   = np.linspace(0, 1, 21)     # 21 uniform bins (paper §4.3)

Observation features (14-dim):
    [0]  time_left          (sim_time - t) / sim_time
    [1]  fundamental        value / N["fundamental"]
    [2]  best_ask           ask / N["fundamental"];  1.0 when empty
    [3]  best_bid           bid / N["fundamental"];  0.0 when empty
    [4]  inventory          clip(position / N["invt"], -1, 1)
    [5]  midprice_delta     clip(midprice_move / 1e2, -1, 1)
    [6]  vol_imbalance      volume_imbalance(market)
    [7]  queue_imbalance    queue_imbalance(market)
    [8]  realized_vol       clip(realized_volatility, 0, 1)
    [9]  rsi                rsi / 100
    [10] est_fundamental    Bayesian estimate / N["fundamental"]
    [11] pv_buy             clip(pv.value_for_exchange(pos, BUY)  / N["pv"], -1, 1)
    [12] pv_sell            clip(pv.value_for_exchange(pos, SELL) / N["pv"], -1, 1)
    [13] side               0.0 = BUY, 1.0 = SELL
"""

import math
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as dist
from gymnasium import spaces

from marketsim.agent.tron_agent import TRONPolicy
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


class TRONEnv(gym.Env):
    """Gymnasium environment for training the TRON recurrent DQN agent.

    The RL agent and all background agents share a single arrival pool — no
    explicit warm-up phase.  Both use Geometric inter-arrival sampling;
    background agents use rate `lam`, the RL agent uses rate `lam_zi`.
    """

    metadata = {"render_modes": []}

    # Discrete action bins (must match TRONPolicy)
    SHADE_BINS: np.ndarray = TRONPolicy.SHADE_BINS
    ETA_BINS = TRONPolicy.ETA_BINS

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
        normalizers=None,
        bg_strategies=None,
        shade_bins=None,
        eta_bins=None,
        advantage_reward: bool = False,
        paired_proxy: bool = False,
    ):
        """
        Args:
            num_background_agents: Number of background ZI traders.
            sim_time:              Total simulation time steps per episode.
            lam:                   Arrival rate for background agents (Geometric).
            lam_zi:                Arrival rate for the RL agent (Geometric).
            mean:                  Long-run fundamental mean.
            r:                     Mean-reversion rate.
            shock_var:             Variance of fundamental shocks.
            q_max:                 Maximum absolute inventory position.
            pv_var:                Variance of private value draws.
            shade:                 Background agents' shade range [low, high].
            normalizers:           Dict with keys "fundamental", "invt",
                                   "reward", "pv" for observation/reward scaling.
            bg_strategies:         Optional list of dicts with 'shade' and 'eta'.
                                   Each reset() picks one randomly for all BG agents.
            paired_proxy:          If True, add a ZI S8 proxy agent with TRON's shared
                                   private value. Reward = (TRON_delta - proxy_delta),
                                   replacing the mean-bg baseline. Reduces pv noise.
        """
        super().__init__()

        if shade is None:
            shade = [250, 500]
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
        self.normalizers = normalizers
        self.bg_strategies = bg_strategies
        self.advantage_reward = advantage_reward
        self.paired_proxy = paired_proxy
        if shade_bins is not None:
            self.SHADE_BINS = np.asarray(shade_bins)
        if eta_bins is not None:
            self.ETA_BINS = np.asarray(eta_bins)
        self.time = 0
        self.last_value = 0.0
        self.final_fundamental: float = 0.0  # realized F_T for this episode (paper §3.5)
        # Running total value of all bg agents at the last TRON step (for advantage reward)
        self.bg_last_total_value = 0.0
        self.current_side = BUY  # side assigned before each RL decision
        # Paired proxy state (Option A)
        self.proxy_agent = None
        self.proxy_agent_id = None
        self.proxy_last_value = 0.0
        self.arrival_times_proxy = None
        self.arrival_index_proxy = 0

        # ── Arrival buffers ───────────────────────────────────────────────
        # All agents (BG + RL) share the same arrivals dict.
        # BG agents are sampled at rate lam; RL agent at rate lam_zi.
        self.arrivals_sampled = 10000

        self.arrivals = defaultdict(list)
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        # Separate inter-arrival buffer for the RL agent (may have different rate)
        self.arrival_times_zi = sample_arrivals(lam_zi, self.arrivals_sampled)
        self.arrival_index_zi = 0

        # ── Market ────────────────────────────────────────────────────────
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

        # ── RL agent — joins the shared arrivals pool ─────────────────────
        self.zi_agent_id = num_background_agents
        first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        self.arrivals[first_zi].append(self.zi_agent_id)
        self.arrival_index_zi += 1
        self.zi_agent = ZIAgent(
            agent_id=self.zi_agent_id,
            market=self.market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
        )

        # ── Paired proxy agent (Option A) ─────────────────────────────────
        if paired_proxy:
            self.proxy_agent_id = num_background_agents + 1  # one beyond TRON
            self.proxy_agent = ZIAgent(
                agent_id=self.proxy_agent_id,
                market=self.market,
                q_max=q_max,
                shade=[460, 540],   # S8 strategy
                pv_var=pv_var,
                eta=0.5,
            )
            self.arrival_times_proxy = sample_arrivals(lam_zi, self.arrivals_sampled)
            self.arrival_index_proxy = 0
            first_proxy = self.arrival_times_proxy[self.arrival_index_proxy].item()
            self.arrivals[first_proxy].append(self.proxy_agent_id)
            self.arrival_index_proxy += 1
            # Add to agents dict so agents_step() picks it up automatically
            self.agents[self.proxy_agent_id] = self.proxy_agent

        # ── Gym spaces ────────────────────────────────────────────────────
        # 14-dim: 13 ZIEnv features + side indicator in [0,1]
        lower_bound = np.array(
            [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0], dtype=np.float64
        )
        upper_bound = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=lower_bound, high=upper_bound, shape=(14,), dtype=np.float64
        )
        self.action_space = spaces.MultiDiscrete(
            [len(self.SHADE_BINS), len(self.ETA_BINS)]
        )
        self.observation = np.zeros(14, dtype=np.float64)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.last_value = 0.0
        self.bg_last_total_value = 0.0
        self.proxy_last_value = 0.0
        self.observation = np.zeros(14, dtype=np.float64)

        fundamental = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.sim_time + 1,
            r=self.r,
            shock_var=self.shock_var,
        )
        fundamental.prefetch_all()
        self.final_fundamental = fundamental.get_final_fundamental()
        self.market.reset(fundamental=fundamental)

        if self.bg_strategies is not None:
            chosen = random.choice(self.bg_strategies)
            for agent_id in self.agents:
                # Don't override proxy's fixed S8 strategy
                if agent_id == self.proxy_agent_id:
                    continue
                self.agents[agent_id].shade = chosen['shade']
                self.agents[agent_id].eta = chosen.get('eta', 1.0)

        for agent_id in self.agents:
            self.agents[agent_id].reset()
        self.zi_agent.reset()

        if self.paired_proxy:
            # Share TRON's private value so pv noise cancels in the reward difference
            self.proxy_agent.pv = self.zi_agent.pv

        # Resample arrival schedules (RL agent goes into same pool — no warm-up)
        self.reset_arrivals()

        end = self.run_until_next_zi_arrival()
        if end:
            raise ValueError(
                "TRONEnv: episode ended before RL agent arrived. "
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
            obs, end_r, term, trunc, info = self.end_sim()
            return obs, reward + end_r, term, trunc, info

        return self.get_obs(), reward, False, False, {}

    def render(self):
        pass

    # ── Arrival helpers ────────────────────────────────────────────────────

    def reset_arrivals(self):
        """Resample all arrival schedules. RL agent joins the shared arrivals pool."""
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrival_times_zi = sample_arrivals(self.lam_zi, self.arrivals_sampled)
        self.arrival_index_zi = 0

        # Schedule background agents' first arrivals
        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(
                agent_id
            )
            self.arrival_index += 1

        # Schedule RL agent's first arrival into the shared pool
        first_zi = self.arrival_times_zi[self.arrival_index_zi].item()
        self.arrivals[first_zi].append(self.zi_agent_id)
        self.arrival_index_zi += 1

        # Schedule proxy agent's first arrival (same rate as TRON)
        if self.paired_proxy:
            self.arrival_times_proxy = sample_arrivals(self.lam_zi, self.arrivals_sampled)
            self.arrival_index_proxy = 0
            first_proxy = self.arrival_times_proxy[self.arrival_index_proxy].item()
            self.arrivals[first_proxy].append(self.proxy_agent_id)
            self.arrival_index_proxy += 1

    def run_until_next_zi_arrival(self):
        """Advance market until the RL agent's next scheduled arrival.

        Returns:
            True  if episode ended (time >= sim_time) before RL agent arrived.
            False if RL agent arrived; also calls update_obs().
        """
        while (
            self.zi_agent_id not in self.arrivals[self.time]
            and self.time < self.sim_time
        ):
            self.agents_step()
            self.market_step(agent_only=True)
            self.time += 1

        if self.time >= self.sim_time:
            return True
        else:
            # Pre-assign side for this RL decision; include in observation
            self.current_side = random.choice([BUY, SELL])
            self.update_obs()
            return False

    # ── Per-step helpers ───────────────────────────────────────────────────

    def zi_agent_step(self, action):
        """Decode discrete action [shade_idx, eta_idx] → limit order.

        Args:
            action: array-like [shade_idx, eta_idx]
        """
        shade_val = float(self.SHADE_BINS[int(action[0])])
        eta = float(self.ETA_BINS[int(action[1])])
        side = self.current_side

        self.market.event_queue.set_time(self.time)
        self.market.withdraw_all(self.zi_agent_id)

        t = self.market.get_time()
        estimate = self.zi_agent.estimate_fundamental()
        pv_value = self.zi_agent.pv.value_for_exchange(self.zi_agent.position, side)

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

        # Schedule next RL agent arrival into the shared arrivals pool
        if self.arrival_index_zi == self.arrivals_sampled:
            self.arrival_times_zi = sample_arrivals(
                self.lam_zi, self.arrivals_sampled
            )
            self.arrival_index_zi = 0
        self.arrivals[
            self.arrival_times_zi[self.arrival_index_zi].item() + 1 + self.time
        ].append(self.zi_agent_id)
        self.arrival_index_zi += 1

    def agents_step(self):
        """Let all background agents that arrive at self.time act.

        Skips the RL agent ID — the RL agent is handled by zi_agent_step().
        """
        agents = [
            a for a in self.arrivals[self.time] if a != self.zi_agent_id
        ]
        if not agents:
            return
        self.market.event_queue.set_time(self.time)
        for agent_id in agents:
            agent = self.agents[agent_id]
            self.market.withdraw_all(agent_id)
            orders = agent.take_action()
            self.market.add_orders(orders)

            # Proxy reschedules at lam_zi rate; bg agents reschedule at lam rate
            if self.paired_proxy and agent_id == self.proxy_agent_id:
                if self.arrival_index_proxy == self.arrivals_sampled:
                    self.arrival_times_proxy = sample_arrivals(self.lam_zi, self.arrivals_sampled)
                    self.arrival_index_proxy = 0
                self.arrivals[
                    self.arrival_times_proxy[self.arrival_index_proxy].item() + 1 + self.time
                ].append(agent_id)
                self.arrival_index_proxy += 1
            else:
                if self.arrival_index == self.arrivals_sampled:
                    self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                    self.arrival_index = 0
                self.arrivals[
                    self.arrival_times[self.arrival_index].item() + 1 + self.time
                ].append(agent_id)
                self.arrival_index += 1

    def market_step(self, agent_only=True):
        """Clear market, update positions, optionally compute reward."""
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
            elif self.paired_proxy and aid == self.proxy_agent_id:
                self.proxy_agent.update_position(qty, cash)
            else:
                self.agents[aid].update_position(qty, cash)

        if not agent_only:
            # Use the episode's realized final fundamental for mark-to-market (paper §3.5).
            # This ensures intermediate rewards sum exactly to the liquidation return.
            # The Bayesian estimate is kept only for the observation feature [10].
            f = self.final_fundamental
            current_value = (
                self.zi_agent.position * f
                + self.zi_agent.cash
                + self.zi_agent.get_pos_value()
            )
            tron_delta = current_value - self.last_value
            self.last_value = current_value

            if self.paired_proxy:
                proxy_value = (
                    self.proxy_agent.position * f
                    + self.proxy_agent.cash
                    + self.proxy_agent.get_pos_value()
                )
                proxy_delta = proxy_value - self.proxy_last_value
                self.proxy_last_value = proxy_value
                reward = (tron_delta - proxy_delta) / self.normalizers["reward"]
            elif self.advantage_reward:
                bg_total = sum(
                    ag.position * f + ag.cash + ag.get_pos_value()
                    for ag in self.agents.values()
                    if ag is not self.proxy_agent
                )
                n_bg = len(self.agents) - (1 if self.paired_proxy else 0)
                mean_bg_delta = (bg_total - self.bg_last_total_value) / max(1, n_bg)
                self.bg_last_total_value = bg_total
                reward = (tron_delta - mean_bg_delta) / self.normalizers["reward"]
            else:
                reward = tron_delta / self.normalizers["reward"]
            return reward

        return 0.0

    def end_sim(self):
        f = self.final_fundamental
        current_value = (
            self.zi_agent.position * f
            + self.zi_agent.cash
            + self.zi_agent.get_pos_value()
        )
        tron_delta = current_value - self.last_value

        if self.paired_proxy:
            proxy_value = (
                self.proxy_agent.position * f
                + self.proxy_agent.cash
                + self.proxy_agent.get_pos_value()
            )
            proxy_delta = proxy_value - self.proxy_last_value
            reward = (tron_delta - proxy_delta) / self.normalizers["reward"]
        elif self.advantage_reward:
            bg_total = sum(
                ag.position * f + ag.cash + ag.get_pos_value()
                for ag in self.agents.values()
                if ag is not self.proxy_agent
            )
            n_bg = len(self.agents) - (1 if self.paired_proxy else 0)
            mean_bg_delta = (bg_total - self.bg_last_total_value) / max(1, n_bg)
            reward = (tron_delta - mean_bg_delta) / self.normalizers["reward"]
        else:
            reward = tron_delta / self.normalizers["reward"]
        return self.get_obs(), reward, True, False, {}

    # ── Observation ────────────────────────────────────────────────────────

    def _env_estimate_fundamental(self) -> float:
        """Bayesian fundamental estimate using self.time (env clock)."""
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
            self.current_side,
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
        side,
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
        side_n = 0.0 if side == BUY else 1.0

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
                side_n,
            ],
            dtype=np.float64,
        )
        if np.isnan(obs[9]):  # rsi_n → neutral
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_obs(self):
        return self.observation

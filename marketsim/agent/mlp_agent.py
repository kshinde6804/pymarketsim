"""
MLPAgent — ZI-style trading agent whose shade/eta are chosen by an internal MLP.

ZIMlpPolicy:
    Simple 3-hidden-layer MLP with Sigmoid output, constraining both actions
    to [0, 1].  Input dim = 14 (normalized market + agent features + side).

MLPAgent:
    Inherits ZIAgent to reuse estimate_fundamental(), update_position(),
    get_pos_value(), reset(), and the PrivateValues object.

    Overrides take_action() to:
        1. Pre-sample side (BUY/SELL).
        2. Build the same 14-feature observation as ZIEnv.update_obs().
        3. Pass it through the internal MLP to get (shade_norm, eta).
        4. Denormalize shade via shade_range.
        5. Use the exact ZI pricing formula from zero_intelligence_agent.py.

    Can be plugged directly into a raw simulation as a drop-in for ZIAgent:
        sim.agents[agent_id] = MLPAgent(agent_id, market, ...)

    Observation feature order (must stay in sync with ZIEnv.normalize()):
        [0]  time_left
        [1]  fundamental
        [2]  best_ask
        [3]  best_bid
        [4]  inventory
        [5]  midprice_delta
        [6]  vol_imbalance
        [7]  queue_imbalance
        [8]  realized_vol
        [9]  rsi
        [10] est_fundamental
        [11] pv_buy
        [12] pv_sell
        [13] side              0.0 = BUY, 1.0 = SELL
"""

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.wrappers.metrics import (
    midprice_move,
    queue_imbalance,
    realized_volatility,
    relative_strength_index,
    volume_imbalance,
)


class ZIMlpPolicy(nn.Module):
    """3-hidden-layer MLP for the ZI RL agent.

    Input:  14 normalized market/agent features (13 market features + side).
    Output: 2 values in [0, 1] via Sigmoid — (shade_norm, eta).
    """

    def __init__(
        self, input_dim: int = 14, hidden_dim: int = 64, output_dim: int = 2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Both outputs constrained to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAgent(ZIAgent):
    """ZI-style trading agent driven by an internal PyTorch MLP.

    Args:
        agent_id:     Unique integer identifier.
        market:       The shared Market object.
        q_max:        Maximum absolute inventory position.
        pv_var:       Variance of private value draws.
        shade:        Shade range passed to parent ZIAgent (not used in
                      take_action but must be a valid 2-element list).
        shade_range:  [d_min, d_max] for denormalizing MLP action[0] →
                      shade = action[0] * (d_max - d_min) + d_min.
        normalizers:  Dict with "fundamental", "invt", "pv" keys matching
                      those used in ZIEnv to ensure feature consistency.
        hidden_dim:   Hidden layer width (default 64).
        weights_path: Path to a saved state_dict (.pt) to load pretrained
                      weights; leave None for a freshly-initialized policy.
    """

    def __init__(
        self,
        agent_id: int,
        market,
        q_max: int,
        pv_var: float,
        shade=None,
        shade_range=None,
        normalizers=None,
        hidden_dim: int = 64,
        weights_path: str = None,
    ):
        if shade is None:
            shade = [250, 500]
        if shade_range is None:
            shade_range = [0, 1000]
        if normalizers is None:
            normalizers = {"fundamental": 1e5, "invt": 10, "pv": 5e5}

        # Parent sets up: pv, position, cash, _order_counter, shade, q_max, etc.
        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
        )

        self.shade_range = shade_range
        self.normalizers = normalizers

        self.policy = ZIMlpPolicy(
            input_dim=14, hidden_dim=hidden_dim, output_dim=2
        )
        self.policy.eval()

        if weights_path is not None:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )

    # ── Observation ───────────────────────────────────────────────────────

    def build_obs(self, side: int) -> np.ndarray:
        """Build the 14-feature observation identical to ZIEnv.update_obs().

        Args:
            side: BUY (1) or SELL (-1) — the side pre-sampled for this action.
                  Encoded as 0.0 for BUY, 1.0 for SELL (matching TRONAgent).

        Uses market.fundamental.final_time to derive sim_time so the agent
        is self-contained (no dependency on the ZIEnv object).
        """
        t = self.market.get_time()
        # LazyGaussianMeanReverting stores final_time = sim_time + 1
        sim_time = self.market.fundamental.final_time - 1

        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask = self.market.order_book.get_best_ask()
        best_bid = self.market.order_book.get_best_bid()
        inventory = self.position

        mp_delta = midprice_move(self.market)
        vol_imb = volume_imbalance(self.market)
        que_imb = queue_imbalance(self.market)
        rv = realized_volatility(self.market)
        rsi = relative_strength_index(self.market)

        est_fund = self.estimate_fundamental()
        pv_buy = self.pv.value_for_exchange(self.position, BUY)
        pv_sell = self.pv.value_for_exchange(self.position, SELL)

        N = self.normalizers
        pv_n = N.get("pv", 5e5)

        time_left_n = (sim_time - t) / sim_time if sim_time > 0 else 0.0
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
        if np.isnan(obs[9]):   # rsi_n
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── Action ────────────────────────────────────────────────────────────

    def take_action(self) -> List[Order]:
        """Override ZIAgent.take_action(): use MLP to select shade and eta.

        Side is pre-sampled before building the observation so that the agent
        can condition shade/eta on which side it will be trading.
        """
        side = random.choice([BUY, SELL])
        obs = self.build_obs(side)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_tensor).squeeze(0).numpy()

        shade_norm, eta = float(action[0]), float(action[1])

        d_min, d_max = self.shade_range
        shade_val = shade_norm * (d_max - d_min) + d_min

        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - shade_val
        else:
            price = estimate + pv_value + shade_val

        # eta: take liquidity if surplus fraction exceeds threshold
        # (exact replica of zero_intelligence_agent.py:54-63)
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

        return [
            Order(
                price=price,
                quantity=1,
                agent_id=self.agent_id,
                time=t,
                order_type=side,
                order_id=order_id,
            )
        ]

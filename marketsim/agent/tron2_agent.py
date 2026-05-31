"""
tron2_agent.py — Brand-new TRON implementation based on ICAIF24 paper (§4.3, Fig. 3).

TRONPolicy2: Dueling DQN + LSTM architecture.
TRONAgent2:  Extends ZIAgent; wraps TRONPolicy2 for use inside the base Simulator.

Architecture (Fig. 3):
    Input (14-dim)
      → Linear(14, 128) → ReLU            [shared encoder]
      → LSTM(128, 128, batch_first=True)   [stateful per episode]
      →  Value head:  Linear(128,128)→ReLU→Linear(128,2)   [V_shade, V_eta]
         Adv   head:  Linear(128,128)→ReLU→Linear(128,42)  [A_shade(21), A_eta(21)]
      → Q_shade = V[...,0:1] + A_shade − mean(A_shade)
         Q_eta  = V[...,1:2] + A_eta   − mean(A_eta)

Action bins (module-level constants, easy to change):
    SHADE_BINS = np.linspace(0, SHADE_MAX, 21)
    ETA_BINS   = np.linspace(0, 1, 21)
"""

import math

import numpy as np
import torch
import torch.nn as nn

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order

# ── Bin constants ──────────────────────────────────────────────────────────────
SHADE_MAX  = 1000                             # upper bound for shade bins
SHADE_BINS = np.linspace(0, SHADE_MAX, 21)   # (21,) uniformly spaced shade values
ETA_BINS   = np.linspace(0, 1, 21)           # (21,) uniformly spaced eta values
N_SHADE    = len(SHADE_BINS)                 # 21
N_ETA      = len(ETA_BINS)                   # 21


class TRONPolicy2(nn.Module):
    """Dueling DQN + LSTM as specified in Fig. 3 of the ICAIF24 paper.

    Inputs:   obs tensor (B, T, input_dim)  — or (1, 1, 14) for single-step inference
    Outputs:  Q_shade (B, T, n_shade=21), Q_eta (B, T, n_eta=21), h_c_new
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        n_shade: int = N_SHADE,
        n_eta: int = N_ETA,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_shade = n_shade
        self.n_eta = n_eta

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Recurrent layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Value stream — one scalar V per action head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # V_shade, V_eta
        )

        # Advantage stream — 21 + 21 = 42 outputs
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_shade + n_eta),
        )

    def forward(self, obs, h_c=None):
        """
        Args:
            obs:  FloatTensor (B, T, input_dim)
            h_c:  optional (h, c) each (1, B, hidden_dim); None → zeros

        Returns:
            Q_shade: (B, T, n_shade)
            Q_eta:   (B, T, n_eta)
            h_c_new: updated LSTM state
        """
        x = self.encoder(obs)               # (B, T, hidden_dim)
        x, h_c_new = self.lstm(x, h_c)     # (B, T, hidden_dim)

        V = self.value_head(x)             # (B, T, 2)
        A = self.advantage_head(x)         # (B, T, n_shade + n_eta)

        A_shade = A[..., : self.n_shade]   # (B, T, n_shade)
        A_eta   = A[..., self.n_shade :]   # (B, T, n_eta)

        V_shade = V[..., 0:1]              # (B, T, 1)
        V_eta   = V[..., 1:2]              # (B, T, 1)

        # Dueling: Q = V + A − mean(A)  (Wang et al. 2016)
        Q_shade = V_shade + A_shade - A_shade.mean(dim=-1, keepdim=True)
        Q_eta   = V_eta   + A_eta   - A_eta.mean(  dim=-1, keepdim=True)

        return Q_shade, Q_eta, h_c_new


class TRONAgent2(ZIAgent):
    """TRON agent for deployment inside the base PyMarketSim Simulator.

    Wraps TRONPolicy2 and implements ZIAgent.take_action().
    The caller must pre-assign self.current_side before each take_action() call
    (mimicking how TRONEnv2 pre-assigns the side before returning an observation).

    Usage inside Simulator:
        agent.current_side = random.choice([BUY, SELL])
        orders = agent.take_action()
    """

    def __init__(
        self,
        agent_id: int,
        market,
        q_max: int,
        pv_var: float,
        policy: TRONPolicy2,
        shade_bins: np.ndarray = SHADE_BINS,
        eta_bins: np.ndarray = ETA_BINS,
        device=None,
    ):
        # ZIAgent base: shade/eta placeholders; overridden by take_action().
        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            shade=[0, 0],
            pv_var=pv_var,
            eta=1.0,
        )
        self.policy     = policy
        self.shade_bins = shade_bins
        self.eta_bins   = eta_bins
        self.device     = device or torch.device("cpu")
        self.h_c        = None    # LSTM hidden state (zeroed at episode start)
        self.current_side = BUY  # pre-assigned by caller before take_action()

    def reset(self):
        super().reset()
        self.h_c = None
        self.current_side = BUY

    def take_action(self):
        """Build 14-dim obs → policy forward → ZI-style limit order."""
        obs = self._build_obs()
        obs_t = (
            torch.tensor(obs, dtype=torch.float32)
            .unsqueeze(0)   # batch dim
            .unsqueeze(0)   # time dim
            .to(self.device)
        )
        with torch.no_grad():
            Q_shade, Q_eta, self.h_c = self.policy(obs_t, self.h_c)

        shade_idx = int(Q_shade.squeeze().argmax().item())
        eta_idx   = int(Q_eta.squeeze().argmax().item())

        shade = float(self.shade_bins[shade_idx])
        eta   = float(self.eta_bins[eta_idx])

        return self._place_order(shade, eta)

    # ── Observation ────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """14-dim observation identical to TRONEnv2.normalize()."""
        from marketsim.wrappers.metrics import (
            midprice_move,
            queue_imbalance,
            realized_volatility,
            relative_strength_index,
            volume_imbalance,
        )

        t = self.market.get_time()
        T = self.market.end_time
        mean, r, _ = self.market.get_info()

        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask = self.market.order_book.get_best_ask()
        best_bid = self.market.order_book.get_best_bid()

        rho      = (1.0 - r) ** (T - t)
        est_fund = (1.0 - rho) * mean + rho * fundamental_value

        pv_buy  = self.pv.value_for_exchange(self.position, BUY)
        pv_sell = self.pv.value_for_exchange(self.position, SELL)

        N_FUND = 1e5
        N_INV  = 10
        N_PV   = 5e5

        time_left_n = (T - t) / T
        fund_n      = fundamental_value / N_FUND
        ask_n       = 1.0 if math.isinf(best_ask) else best_ask / N_FUND
        bid_n       = 0.0 if math.isinf(best_bid) else best_bid / N_FUND
        inv_n       = float(np.clip(self.position / N_INV, -1.0, 1.0))
        mp_n        = float(np.clip(midprice_move(self.market) / 1e2, -1.0, 1.0))
        vol_imb     = float(volume_imbalance(self.market))
        que_imb     = float(queue_imbalance(self.market))
        rv          = float(np.clip(realized_volatility(self.market), 0.0, 1.0))
        rsi_n       = float(relative_strength_index(self.market)) / 100.0
        est_n       = est_fund / N_FUND
        pv_buy_n    = float(np.clip(pv_buy  / N_PV, -1.0, 1.0))
        pv_sell_n   = float(np.clip(pv_sell / N_PV, -1.0, 1.0))
        side_n      = 0.0 if self.current_side == BUY else 1.0

        obs = np.array(
            [
                time_left_n, fund_n, ask_n, bid_n, inv_n,
                mp_n, vol_imb, que_imb, rv, rsi_n,
                est_n, pv_buy_n, pv_sell_n, side_n,
            ],
            dtype=np.float64,
        )
        if np.isnan(obs[9]):   # RSI defaults to neutral (50/100) when no data
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── Order placement ────────────────────────────────────────────────────────

    def _place_order(self, shade: float, eta: float):
        """ZI-style limit order with a specific shade value (not a drawn range).

        Mirrors ZIAgent.take_action() exactly, substituting the drawn random
        offset with the TRON-selected shade value (paper §4.3).
        """
        t        = self.market.get_time()
        side     = self.current_side
        estimate = self.estimate_fundamental()
        pv_value = self.pv.value_for_exchange(self.position, side)

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

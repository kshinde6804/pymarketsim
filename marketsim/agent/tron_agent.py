"""
TRONAgent — Trained Recurrent Order Network agent (ICAIF24).

Uses a Dueling DQN architecture with an LSTM recurrent component to select
discrete shade and eta actions, then applies the standard ZI pricing formula.

Architecture:
    Input: 14-dim (13 ZIEnv features + side indicator)
    Shared encoder: Linear(14→128) → ReLU
    Recurrent: LSTM(128, 128, batch_first=True)
    Value head: Linear(128→1)  [shared scalar V(s)]
    Advantage shade: Linear(128→128) → ReLU → Linear(128→42)
    Advantage eta:   Linear(128→128) → ReLU → Linear(128→2)
    Q_shade = V + A_shade − mean(A_shade)  →  42 discrete shade values
    Q_eta   = V + A_eta   − mean(A_eta)    →  2 discrete eta values

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
    [9]  rsi                relative_strength_index / 100
    [10] est_fundamental    Bayesian estimate / N["fundamental"]
    [11] pv_buy             clip(pv.value_for_exchange(pos, BUY)  / N["pv"], -1, 1)
    [12] pv_sell            clip(pv.value_for_exchange(pos, SELL) / N["pv"], -1, 1)
    [13] side               0.0 = BUY, 1.0 = SELL
"""

import math
import random
from typing import List, Optional, Tuple

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


class TRONPolicy(nn.Module):
    """Dueling DQN + LSTM policy for the TRON agent.

    Args:
        input_dim:    Observation dimension (default 14).
        hidden_dim:   Encoder and LSTM hidden size (default 128).
        n_shade_bins: Number of discrete shade actions (default 42).
        n_eta_bins:   Number of discrete eta actions (default 2).
    """

    SHADE_BINS: np.ndarray = np.linspace(0, 600, 42)  # 42 uniformly spaced values
    ETA_BINS: List[float] = [0.0, 1.0]                 # binary: market-take vs market-make

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        n_shade_bins: int = 42,
        n_eta_bins: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Shared value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Advantage streams
        self.adv_shade = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_shade_bins),
        )
        self.adv_eta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_eta_bins),
        )

    def forward(
        self,
        x: torch.Tensor,
        h_c: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through encoder → LSTM → dueling heads.

        Args:
            x:   Input tensor of shape (batch, seq_len, input_dim) or
                 (batch, input_dim) — single-step inference adds seq_len=1.
            h_c: Tuple (h, c) of LSTM hidden states, each (1, batch, hidden_dim).
                 If None, initializes to zeros.

        Returns:
            Q_shade: (batch, seq_len, n_shade_bins)
            Q_eta:   (batch, seq_len, n_eta_bins)
            (h, c):  Updated LSTM states.
        """
        # Handle 2-D input (batch, input_dim) → (batch, 1, input_dim)
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Encode each time step
        batch, seq_len, _ = x.shape
        enc = self.encoder(x.reshape(batch * seq_len, -1))
        enc = enc.reshape(batch, seq_len, -1)

        # LSTM
        lstm_out, (h, c) = self.lstm(enc, h_c)  # (batch, seq_len, hidden_dim)

        # Dueling: V + (A - mean(A))
        V = self.value_head(lstm_out)  # (batch, seq_len, 1)

        A_shade = self.adv_shade(lstm_out)  # (batch, seq_len, n_shade)
        A_eta = self.adv_eta(lstm_out)      # (batch, seq_len, n_eta)

        Q_shade = V + A_shade - A_shade.mean(dim=-1, keepdim=True)
        Q_eta = V + A_eta - A_eta.mean(dim=-1, keepdim=True)

        if squeeze:
            Q_shade = Q_shade.squeeze(1)  # (batch, n_shade)
            Q_eta = Q_eta.squeeze(1)      # (batch, n_eta)

        return Q_shade, Q_eta, (h, c)


class TRONAgent(ZIAgent):
    """ZI-style trading agent driven by a recurrent Dueling DQN (TRON).

    Drop-in replacement for ZIAgent / MLPAgent in a raw Simulator or
    equilibrium experiment.  LSTM state is stored per-episode and zeroed
    on reset().

    Args:
        agent_id:     Unique integer identifier.
        market:       The shared Market object.
        q_max:        Maximum absolute inventory position.
        pv_var:       Variance of private value draws.
        shade:        Shade range passed to parent ZIAgent (not used in
                      take_action but required by ZIAgent.__init__).
        normalizers:  Dict with "fundamental", "invt", "pv" keys.
        weights_path: Path to a saved TRONPolicy state_dict (.pt).
                      Leave None for a freshly-initialized (random) policy.
    """

    def __init__(
        self,
        agent_id: int,
        market,
        q_max: int,
        pv_var: float,
        shade=None,
        normalizers=None,
        weights_path: Optional[str] = None,
    ):
        if shade is None:
            shade = [250, 500]
        if normalizers is None:
            normalizers = {"fundamental": 1e5, "invt": 10, "pv": 5e5}

        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
        )

        self.normalizers = normalizers

        self.policy = TRONPolicy(input_dim=14)
        self.policy.eval()

        if weights_path is not None:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )

        # LSTM state: (h, c) each of shape (1, 1, hidden_dim)
        self._reset_lstm()

    # ── LSTM state helpers ────────────────────────────────────────────────

    def _reset_lstm(self):
        """Zero out LSTM hidden and cell states."""
        h = torch.zeros(1, 1, self.policy.hidden_dim)
        c = torch.zeros(1, 1, self.policy.hidden_dim)
        self.h_c = (h, c)

    def reset(self):
        """Reset position/cash/PV (via super) and zero LSTM state."""
        super().reset()
        self._reset_lstm()

    # ── Observation ───────────────────────────────────────────────────────

    def build_obs(self, side: int) -> np.ndarray:
        """Build the 14-feature observation (13 market features + side).

        Uses market.end_time as sim_time so the agent works correctly with
        both GaussianMeanReverting and LazyGaussianMeanReverting fundamentals.

        Args:
            side: BUY or SELL constant from fourheap.constants.

        Returns:
            14-dim float64 array.
        """
        t = self.market.get_time()
        sim_time = self.market.end_time  # correct for both fundamental types

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
        # Guard against NaN/inf (sparse book, no history)
        if np.isnan(obs[9]):  # rsi_n → neutral
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── Action ────────────────────────────────────────────────────────────

    def take_action(self) -> List[Order]:
        """Select discrete shade+eta via recurrent DQN, then apply ZI pricing."""
        side = random.choice([BUY, SELL])
        obs = self.build_obs(side)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 14)
        with torch.no_grad():
            Q_shade, Q_eta, self.h_c = self.policy(obs_tensor, self.h_c)

        shade_idx = int(Q_shade.argmax(dim=-1).item())
        eta_idx = int(Q_eta.argmax(dim=-1).item())

        shade_val = float(TRONPolicy.SHADE_BINS[shade_idx])
        eta = float(TRONPolicy.ETA_BINS[eta_idx])

        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - shade_val
        else:
            price = estimate + pv_value + shade_val

        # eta: take liquidity if surplus fraction exceeds threshold
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

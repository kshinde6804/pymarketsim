"""
TRONformerAgent — Pre-LN Transformer-based trading agent (ICAIF'24).

Replaces TRON's LSTM backbone with a Pre-LN multi-head self-attention transformer
following Xiong et al. (2020).  The agent maintains a rolling observation buffer
(no recurrent state) and attends causally over past observations.

Architecture (§4.1):
    Input projection:  Linear(14 → d_model=128)
    [CLS] token:       learnable nn.Parameter prepended to the sequence
    Positional enc:    Sinusoidal (paper uses rotary [32]; sinusoidal approximation)
    N=2 Pre-LN blocks: x = x + MHA(LN(x))   [NO causal mask — encoder-only]
                       x = x + FFN(LN(x))    [ffn_hidden=512=4×d_model, h=8 heads]
    Dueling heads:     V: Linear(128→1)
                       Adv shade: Linear(128→128)→ReLU→Linear(128→42)
                       Adv eta:   Linear(128→128)→ReLU→Linear(128→2)
                       Q = V + A − mean(A)  (independently for shade and eta)

NOTE — Remaining paper discrepancy (deferred):
    Paper §4.1: "h_CLS is used to calculate the policy." The correct
    implementation computes Q-values from the CLS token output (h[:,0,:])
    rather than the last sequence position (h[:,-1,:]).  Doing so requires
    a training-algorithm refactor (single Q-value per sequence instead of
    per-timestep Q-values) and is left for future work.

Observation features (14-dim, identical to TRONAgent):
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
    [13] side  (0.0 = BUY, 1.0 = SELL)
"""

import collections
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

# Rolling context window — must match SEQ_LEN in train_tronformer.py
SEQ_LEN: int = 40


# ── Positional Encoding ───────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Model (embedding) dimension.
        max_len: Maximum sequence length supported.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]


# ── Pre-LN Transformer Block ──────────────────────────────────────────────────


class PreLNTransformerBlock(nn.Module):
    """Single Pre-LN transformer block following Xiong et al. (2020).

    Applies LayerNorm *before* the sub-layers (pre-norm):
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))

    Args:
        d_model:    Model dimension (default 128).
        n_heads:    Number of attention heads (default 8, head_dim=16).
        ffn_hidden: FFN inner dimension (default 512 = 4 × d_model).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, ffn_hidden: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one Pre-LN transformer block.

        Args:
            x:         (batch, seq_len, d_model)
            attn_mask: Additive causal mask (seq_len, seq_len) with 0 / -inf.

        Returns:
            (batch, seq_len, d_model)
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ── TRONformerPolicy ──────────────────────────────────────────────────────────


class TRONformerPolicy(nn.Module):
    """Dueling DQN + Pre-LN Transformer policy for the TRONformer agent.

    Args:
        input_dim:    Observation dimension (default 14).
        d_model:      Transformer hidden / embedding size (default 128).
        n_heads:      Number of attention heads (default 8, head_dim=16 per paper §4.1).
        n_layers:     Number of Pre-LN transformer blocks (default 2, per paper §4.1).
        ffn_hidden:   FFN inner dimension (default 512 = 4×d_model, per paper §4.1).
        n_shade_bins: Discrete shade actions (default 42).
        n_eta_bins:   Discrete eta actions (default 2).
        shade_bins:   Optional custom shade bin array.
    """

    SHADE_BINS: np.ndarray = np.linspace(0, 600, 42)
    ETA_BINS: List[float] = [0.0, 1.0]

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_hidden: int = 512,
        n_shade_bins: int = 42,
        n_eta_bins: int = 2,
        shade_bins: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.d_model = d_model
        if shade_bins is not None:
            self.SHADE_BINS = np.asarray(shade_bins)
            n_shade_bins = len(self.SHADE_BINS)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal positional encoding (applied after CLS prepend)
        self.pos_enc = PositionalEncoding(d_model)

        # N Pre-LN transformer blocks
        self.blocks = nn.ModuleList(
            [PreLNTransformerBlock(d_model, n_heads, ffn_hidden) for _ in range(n_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Dueling heads
        self.value_head = nn.Linear(d_model, 1)
        self.adv_shade = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_shade_bins),
        )
        self.adv_eta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_eta_bins),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through input projection → transformer → CLS dueling heads.

        Architecture (paper §4.1):
          1. Linear-project observations to d_model.
          2. Prepend learnable [CLS] token.
          3. Apply sinusoidal positional encoding.
          4. Pass through N Pre-LN transformer blocks with a modified causal mask:
               - CLS (row 0): attends to ALL tokens — global context aggregator.
               - Obs token t (row t+1): causal — attends to CLS + obs tokens 0..t-1.
             This ensures obs tokens never see future market states, while CLS
             accumulates the full rolling context.
          5. Policy derived from h_CLS = h[:, 0, :] via dueling heads.

        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim) for single step.

        Returns:
            Q_shade: (batch, n_shade_bins)  — Q-values from CLS representation
            Q_eta:   (batch, n_eta_bins)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, seq_len, _ = x.shape

        # Project observations to d_model
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token → (batch, seq_len+1, d_model)
        cls = self.cls_token.expand(batch, 1, self.d_model)
        h = torch.cat([cls, h], dim=1)

        # Positional encoding
        h = self.pos_enc(h)

        # Modified causal mask:
        #   - Row 0 (CLS): no masking — CLS attends to every token.
        #   - Rows 1..T (obs): standard upper-triangular causal mask so obs_t
        #     attends only to CLS + obs tokens at positions ≤ t.
        total_len = seq_len + 1
        attn_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn_mask[0, :] = 0.0  # CLS row: attend to all

        # Transformer blocks
        for block in self.blocks:
            h = block(h, attn_mask)

        h = self.final_norm(h)

        # Policy from CLS token (position 0) — paper §4.1: "h_CLS is used to
        # calculate the policy of the TRONformer."
        h_cls = h[:, 0, :]  # (batch, d_model)

        # Dueling: Q = V + A − mean(A)
        V       = self.value_head(h_cls)       # (batch, 1)
        A_shade = self.adv_shade(h_cls)        # (batch, n_shade_bins)
        A_eta   = self.adv_eta(h_cls)          # (batch, n_eta_bins)

        Q_shade = V + A_shade - A_shade.mean(dim=-1, keepdim=True)  # (batch, n_shade_bins)
        Q_eta   = V + A_eta   - A_eta.mean(dim=-1, keepdim=True)    # (batch, n_eta_bins)

        return Q_shade, Q_eta


# ── TRONformerAgent ───────────────────────────────────────────────────────────


class TRONformerAgent(ZIAgent):
    """ZI-style trading agent driven by a Pre-LN Transformer Dueling DQN.

    Maintains a rolling observation buffer (``obs_buffer``) of the last
    ``seq_len`` observations.  At each ``take_action()`` call the full buffer
    is stacked into a sequence and passed through ``TRONformerPolicy``; the
    Q-values at the *last* position determine the action.

    Args:
        agent_id:     Unique integer identifier.
        market:       The shared Market object.
        q_max:        Maximum absolute inventory position.
        pv_var:       Variance of private value draws.
        shade:        Shade range passed to parent ZIAgent.
        normalizers:  Dict with "fundamental", "invt", "pv" keys.
        weights_path: Path to a saved TRONformerPolicy state_dict (.pt).
                      Leave None for a freshly-initialized (random) policy.
        seq_len:      Rolling context window size (default SEQ_LEN=32).
        d_model:      Transformer hidden size (default 128).
        shade_bins:   Optional custom shade bin array.
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
        seq_len: int = SEQ_LEN,
        d_model: int = 128,
        shade_bins: Optional[np.ndarray] = None,
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
        self.seq_len = seq_len

        self.policy = TRONformerPolicy(
            input_dim=14, d_model=d_model, shade_bins=shade_bins
        )
        self.policy.eval()

        if weights_path is not None:
            self.policy.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )

        # Rolling observation buffer (no recurrent state needed)
        self.obs_buffer: collections.deque = collections.deque(maxlen=seq_len)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset position/cash/PV (via super) and clear the observation buffer."""
        super().reset()
        self.obs_buffer.clear()

    # ── Observation ───────────────────────────────────────────────────────────

    def build_obs(self, side: int) -> np.ndarray:
        """Build the 14-feature observation (13 market features + side indicator).

        Identical to TRONAgent.build_obs().  Uses market.end_time as sim_time
        so the agent works correctly with both GaussianMeanReverting and
        LazyGaussianMeanReverting fundamentals.

        Args:
            side: BUY or SELL constant from fourheap.constants.

        Returns:
            14-dim float64 array.
        """
        t = self.market.get_time()
        sim_time = self.market.end_time

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
        if np.isnan(obs[9]):
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    def take_action(self) -> List[Order]:
        """Select discrete shade+eta via transformer DQN, then apply ZI pricing."""
        side = random.choice([BUY, SELL])
        obs = self.build_obs(side)

        # Append to rolling buffer and build sequence tensor
        self.obs_buffer.append(obs.astype(np.float32))
        obs_seq = np.stack(list(self.obs_buffer))  # (cur_len, 14)
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0)  # (1, cur_len, 14)

        with torch.no_grad():
            Q_shade, Q_eta = self.policy(obs_t)  # (1, cur_len, 42), (1, cur_len, 2)

        # Q-values from CLS token — shape (1, n_bins)
        shade_idx = int(Q_shade[0, :].argmax().item())
        eta_idx   = int(Q_eta[0, :].argmax().item())

        shade_val = float(self.policy.SHADE_BINS[shade_idx])
        eta = float(TRONformerPolicy.ETA_BINS[eta_idx])

        t = self.market.get_time()
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

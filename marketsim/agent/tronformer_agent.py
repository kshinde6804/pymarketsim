"""
TRONformerAgent — Pre-LN Transformer-based trading agent (ICAIF'24).

Replaces TRON's LSTM backbone with a Pre-LN multi-head self-attention transformer
following Xiong et al. (2020).  The agent maintains a rolling observation buffer
(no recurrent state) and attends causally over past observations.

Architecture (§4.1):
    Input projection:  Linear(14 → d_model=128)
    Positional enc:    Rotary Position Embeddings (RoPE) [Su et al., 2021]
                       applied to Q and K inside each attention block.
                       Observation tokens at positions 0..seq_len-1; most recent
                       obs is always at position seq_len-1 (largest RoPE distance
                       from position 0 gives correct recency bias in causal attn).
    N=2 Pre-LN blocks: x = x + MHA_RoPE(LN(x))   [standard causal mask; token t
                                                    attends to tokens 0..t only]
                       x = x + FFN(LN(x))          [ffn_hidden=512=4×d_model, h=8 heads]
    Dueling heads:     V: Linear(128→1)
                       Adv shade: Linear(128→128)→ReLU→Linear(128→42)
                       Adv eta:   Linear(128→128)→ReLU→Linear(128→2)
                       Q = V + A − mean(A)  (independently for shade and eta)
                       Applied at ALL positions; inference reads Q[:, -1, :] (last).

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


# ── Rotary Position Embeddings (RoPE) ────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split x along last dim and rotate: [x1, x2] → [−x2, x1]."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE rotation to x using pre-computed cos/sin tables.

    Args:
        x:   (..., seq_len, d_head)
        cos: (seq_len, d_head) — broadcast-compatible
        sin: (seq_len, d_head)

    Returns:
        (..., seq_len, d_head)
    """
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021 — RoFormer).

    Pre-computes cos/sin tables for positions 0..max_len−1.

    Args:
        d_head:  Per-head dimension (d_model // n_heads); must be even.
        max_len: Maximum sequence length to pre-compute.
        base:    Frequency base (default 10 000, per the original paper).
    """

    def __init__(self, d_head: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        half = d_head // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len, d_head)

    def _build_cache(self, max_len: int, d_head: int) -> None:
        positions = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
        freqs = torch.outer(positions, self.inv_freq)            # (max_len, d_head//2)
        emb = torch.cat([freqs, freqs], dim=-1)                  # (max_len, d_head)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for positions 0..seq_len−1.

        Returns:
            cos: (seq_len, d_head)
            sin: (seq_len, d_head)
        """
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


# ── Multi-head Self-Attention with RoPE ──────────────────────────────────────


class MultiheadSelfAttentionWithRoPE(nn.Module):
    """Multi-head self-attention that applies Rotary Position Embeddings to Q and K.

    Replaces nn.MultiheadAttention so that RoPE is injected directly into the
    query and key projections before computing scaled dot-product attention.

    Args:
        d_model: Model dimension (must be divisible by n_heads).
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with RoPE-rotated Q and K.

        Args:
            x:            (batch, seq_len, d_model)
            cos:          (seq_len, d_head) — pre-computed by RotaryEmbedding
            sin:          (seq_len, d_head)
            attn_mask:    (seq_len, seq_len) additive mask (0 / −inf), optional.
            key_pad_mask: (batch, 1, 1, seq_len) additive mask (0 / −inf) that
                          blocks attention to zero-padded key positions, optional.

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # Fused QKV projection then split
        qkv = self.qkv_proj(x)                                   # (B, T, 3·D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                              # each (B, T, H, d_head)
        q = q.transpose(1, 2)                                    # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K (broadcast over batch and head dims)
        cos_ = cos.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, d_head)
        sin_ = sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rotary_emb(q, cos_, sin_)
        k = _apply_rotary_emb(k, cos_, sin_)

        # Scaled dot-product attention
        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale    # (B, H, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask                          # broadcast over B, H
        if key_pad_mask is not None:
            scores = scores + key_pad_mask                       # broadcast over H, T_q
        weights = scores.softmax(dim=-1)
        out = torch.matmul(weights, v)                           # (B, H, T, d_head)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, D)               # (B, T, D)
        return self.out_proj(out)


# ── Pre-LN Transformer Block ──────────────────────────────────────────────────


class PreLNTransformerBlock(nn.Module):
    """Single Pre-LN transformer block following Xiong et al. (2020).

    Applies LayerNorm *before* the sub-layers (pre-norm):
        x = x + MHA_RoPE(LN(x))
        x = x + FFN(LN(x))

    Attention uses MultiheadSelfAttentionWithRoPE, which applies Rotary
    Position Embeddings to Q and K before computing attention scores.

    Args:
        d_model:    Model dimension (default 128).
        n_heads:    Number of attention heads (default 8, head_dim=16).
        ffn_hidden: FFN inner dimension (default 512 = 4 × d_model).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, ffn_hidden: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one Pre-LN transformer block.

        Args:
            x:            (batch, seq_len, d_model)
            cos:          (seq_len, d_head) — RoPE cosine table from RotaryEmbedding
            sin:          (seq_len, d_head) — RoPE sine table from RotaryEmbedding
            attn_mask:    Additive causal mask (seq_len, seq_len) with 0 / -inf.
            key_pad_mask: (batch, 1, 1, seq_len) additive mask blocking padded keys.

        Returns:
            (batch, seq_len, d_model)
        """
        normed = self.norm1(x)
        attn_out = self.attn(normed, cos, sin, attn_mask=attn_mask, key_pad_mask=key_pad_mask)
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

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Rotary positional embeddings — shared across all blocks.
        # d_head = d_model // n_heads = 128 // 8 = 16.
        # Positions 0..seq_len-1; most recent obs at seq_len-1 so it has the
        # correct recency ordering in causal attention (recent = small distance).
        self.rope = RotaryEmbedding(d_head=d_model // n_heads)

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
        """Forward pass through input projection → transformer → per-position dueling heads.

        Returns Q-values at every sequence position.
          - Inference: read Q[:, -1, :] (most recent position, full causal context).
          - Per-sequence training: read Q[:, burnin:burnin+L, :] over the learning
            window.

        RoPE positions 0..seq_len-1. With left-padded sequences the most recent
        obs is always at position seq_len-1.  Token t attends causally to tokens
        0..t; for token seq_len-1 (current step) the RoPE angular distance to
        token seq_len-2 is 1, to token 0 is seq_len-1.  This gives the correct
        recency bias: recent observations have smaller angular distance → higher
        relative attention weight.

        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim) for single step.

        Returns:
            Q_shade: (batch, seq_len, n_shade_bins)
            Q_eta:   (batch, seq_len, n_eta_bins)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, seq_len, _ = x.shape

        # Key-padding mask: left-padded sequences have all-zero rows for padding.
        # input_proj has a bias, so zero inputs project to non-zero d_model vectors;
        # prevent real tokens from attending to these padded positions.
        pad_mask = (x.abs().sum(dim=-1) == 0)          # (batch, seq_len), bool
        key_pad_additive = torch.zeros(
            batch, 1, 1, seq_len, device=x.device
        )                                               # (batch, 1, 1, seq_len)
        key_pad_additive = key_pad_additive.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        # Project observations to d_model
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # RoPE: positions 0..seq_len-1.
        cos, sin = self.rope(seq_len, x.device)  # each (seq_len, d_head)

        # Standard causal mask: token t attends only to tokens 0..t.
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask, key_pad_mask=key_pad_additive)

        h = self.final_norm(h)  # (batch, seq_len, d_model)

        # Dueling heads at ALL positions.
        # Q = V + A − mean(A), applied independently for shade and eta.
        V       = self.value_head(h)   # (batch, seq_len, 1)
        A_shade = self.adv_shade(h)    # (batch, seq_len, n_shade_bins)
        A_eta   = self.adv_eta(h)      # (batch, seq_len, n_eta_bins)

        Q_shade = V + A_shade - A_shade.mean(dim=-1, keepdim=True)
        Q_eta   = V + A_eta   - A_eta.mean(dim=-1, keepdim=True)

        return Q_shade, Q_eta  # (batch, seq_len, n_shade), (batch, seq_len, n_eta)


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

        # Append to rolling buffer and build left-padded sequence tensor.
        # Training stores sequences left-padded with zeros to seq_len via
        # _pad_obs_seq(); inference must match that format.
        self.obs_buffer.append(obs.astype(np.float32))
        obs_raw = np.stack(list(self.obs_buffer))  # (cur_len, 14)
        obs_seq = np.zeros((self.seq_len, 14), dtype=np.float32)
        obs_seq[self.seq_len - len(obs_raw):] = obs_raw
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 14)

        with torch.no_grad():
            Q_shade, Q_eta = self.policy(obs_t)  # (1, seq_len, n_shade), (1, seq_len, n_eta)

        # Q-values from most recent position (seq_len-1) — full causal context
        shade_idx = int(Q_shade[0, -1, :].argmax().item())
        eta_idx   = int(Q_eta[0, -1, :].argmax().item())

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

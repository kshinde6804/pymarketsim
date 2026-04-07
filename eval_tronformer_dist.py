"""
eval_tronformer_dist.py — Run many episodes and report per-episode profit distribution.

Usage:
    python -u eval_tronformer_dist.py \
        --model runs/tronformer_ne24_s8_v2/best_model.pt \
        --num-runs 2000 --n-bg 24 --bg-strategy 8 --skew-bins \
        --output results/equilibrium/tronformer_ne24_s8_v2/dist_2000runs.csv
"""

import argparse
import collections
import multiprocessing as mp
import os

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from marketsim.agent.tronformer_agent import (
    TRONformerPolicy, SEQ_LEN, RotaryEmbedding, PreLNTransformerBlock,
)
from marketsim.wrappers.tron_env import TRONEnv


# ── RoPE + CLS architecture — for checkpoints between RoPE migration and CLS removal ──

class TRONformerPolicyCLS(nn.Module):
    """TRONformerPolicy with RoPE + CLS token — matches checkpoints from commits
    0c40e19 through d9c9307 (after sinusoidal PE was replaced, before CLS was removed)."""

    def __init__(self, input_dim=14, d_model=128, n_heads=8, n_layers=2,
                 ffn_hidden=512, shade_bins=None, n_shade_bins=42, n_eta_bins=2):
        super().__init__()
        self.d_model = d_model
        if shade_bins is not None:
            n_shade_bins = len(shade_bins)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.rope = RotaryEmbedding(d_head=d_model // n_heads)
        self.blocks = nn.ModuleList(
            [PreLNTransformerBlock(d_model, n_heads, ffn_hidden) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)
        self.adv_shade = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_shade_bins)
        )
        self.adv_eta = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_eta_bins)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq_len, _ = x.shape
        pad_mask = (x.abs().sum(dim=-1) == 0)
        cls_not_pad = pad_mask.new_zeros(batch, 1)
        key_pad_mask_bool = torch.cat([cls_not_pad, pad_mask], dim=1)
        key_pad_additive = torch.zeros(batch, 1, 1, seq_len + 1, device=x.device)
        key_pad_additive = key_pad_additive.masked_fill(
            key_pad_mask_bool.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        h = self.input_proj(x)
        cls = self.cls_token.expand(batch, 1, self.d_model)
        h = torch.cat([cls, h], dim=1)
        total_len = seq_len + 1
        cos, sin = self.rope(total_len, x.device)
        attn_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=x.device), diagonal=1
        )
        attn_mask[0, :] = 0.0
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask, key_pad_mask=key_pad_additive)
        h = self.final_norm(h)
        h_cls = h[:, 0, :]
        V = self.value_head(h_cls)
        A_shade = self.adv_shade(h_cls)
        A_eta = self.adv_eta(h_cls)
        Q_shade = V + A_shade - A_shade.mean(dim=-1, keepdim=True)
        Q_eta = V + A_eta - A_eta.mean(dim=-1, keepdim=True)
        return Q_shade, Q_eta


# ── Legacy (sinusoidal PE) architecture — for checkpoints before RoPE migration ──

class _LegacyPositionalEncoding(nn.Module):
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

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class _LegacyPreLNBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, ffn_hidden=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.ReLU(), nn.Linear(ffn_hidden, d_model)
        )

    def forward(self, x, attn_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TRONformerPolicyLegacy(nn.Module):
    """TRONformerPolicy with sinusoidal PE — compatible with pre-RoPE checkpoints."""

    def __init__(self, input_dim=14, d_model=128, n_heads=8, n_layers=2,
                 ffn_hidden=512, shade_bins=None, n_shade_bins=42, n_eta_bins=2):
        super().__init__()
        self.d_model = d_model
        if shade_bins is not None:
            n_shade_bins = len(shade_bins)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _LegacyPositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [_LegacyPreLNBlock(d_model, n_heads, ffn_hidden) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)
        self.adv_shade = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_shade_bins)
        )
        self.adv_eta = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_eta_bins)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq_len, _ = x.shape
        h = self.input_proj(x)
        cls = self.cls_token.expand(batch, 1, self.d_model)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        total_len = seq_len + 1
        attn_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn_mask[0, :] = 0.0
        for block in self.blocks:
            h = block(h, attn_mask)
        h = self.final_norm(h)
        h_cls = h[:, 0, :]
        V = self.value_head(h_cls)
        A_shade = self.adv_shade(h_cls)
        A_eta = self.adv_eta(h_cls)
        Q_shade = V + A_shade - A_shade.mean(dim=-1, keepdim=True)
        Q_eta = V + A_eta - A_eta.mean(dim=-1, keepdim=True)
        return Q_shade, Q_eta

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)
SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

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

ENV = {
    'lam': 0.005, 'mean': 1e5, 'r': 0.01, 'shock_var': 1e6,
    'pv_var': 5e6, 'q_max': 10, 'sim_time': 2000,
}
NORMALIZERS = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}


def _worker(args):
    """Run a chunk of episodes; return list of (tron_profit, mean_bg_profit) per episode."""
    (weights_path, n_runs, n_bg, bg_strategy, lam_zi, n_layers,
     shade_bins, seed, lam, shock_var, pv_var, legacy) = args

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(1 << 31)))

    bg = STRATEGIES[bg_strategy]
    sd = torch.load(weights_path, map_location="cpu")
    has_cls = "cls_token" in sd
    has_rope = "rope.inv_freq" in sd
    if legacy or (has_cls and not has_rope):
        PolicyCls = TRONformerPolicyLegacy          # sinusoidal PE + CLS (pre-RoPE)
    elif has_cls and has_rope:
        PolicyCls = TRONformerPolicyCLS             # RoPE + CLS (intermediate)
    else:
        PolicyCls = TRONformerPolicy                # RoPE, no CLS (current)
    # Auto-detect ffn_hidden from checkpoint (may differ from current default 512)
    ffn_hidden = int(sd["blocks.0.ffn.0.weight"].shape[0]) if "blocks.0.ffn.0.weight" in sd else 512
    policy = PolicyCls(input_dim=14, n_layers=n_layers, shade_bins=shade_bins, ffn_hidden=ffn_hidden)
    policy.load_state_dict(sd)
    policy.eval()

    env_kwargs = dict(
        num_background_agents=n_bg,
        sim_time=ENV["sim_time"],
        lam=lam,
        lam_zi=lam_zi,
        mean=ENV["mean"],
        r=ENV["r"],
        shock_var=shock_var,
        q_max=ENV["q_max"],
        pv_var=pv_var,
        shade=[250, 500],
        normalizers=NORMALIZERS,
        bg_strategies=[{'shade': bg['shade'], 'eta': bg['eta']}],
        warmup_fraction=0.0,
        shade_bins=shade_bins,
    )
    env = TRONEnv(**env_kwargs)

    results = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        obs_buf: collections.deque = collections.deque(maxlen=SEQ_LEN)
        ep_reward = 0.0
        done = False
        while not done:
            obs_buf.append(obs.astype(np.float32))
            obs_t = torch.tensor(
                np.stack(list(obs_buf)), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)
            # Architecture returns (batch, 1+seq_len, bins); read [CLS] at position 0.
            q_s = Q_shade[0, 0]
            q_e = Q_eta[0, 0]
            shade_idx = int(q_s.argmax().item())
            eta_idx   = int(q_e.argmax().item())
            obs, r, terminated, truncated, _ = env.step(
                np.array([shade_idx, eta_idx])
            )
            ep_reward += r
            done = terminated or truncated
        tron_profit = ep_reward * NORMALIZERS["reward"]

        # Compute mean bg agent profit in this same episode.
        # end_sim() calls get_final_fundamental() = get_value_at(sim_time+1), which sets
        # latest_t=sim_time+1 and caches that value. Asking for sim_time instead would
        # trigger a backward generation (dt<0). Use sim_time+1 — same value the RL
        # agent's terminal reward was computed against.
        final_fund = env.market.fundamental.get_value_at(env.sim_time + 1)
        bg_profits = [
            agent.position * final_fund + agent.cash + agent.get_pos_value()
            for agent in env.agents.values()
        ]
        mean_bg_profit = float(np.mean(bg_profits))

        results.append((tron_profit, mean_bg_profit))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-runs", type=int, default=2000)
    p.add_argument("--n-bg", type=int, default=24)
    p.add_argument("--bg-strategy", type=int, default=8)
    p.add_argument("--lam-zi", type=float, default=0.02)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--skew-bins", action="store_true")
    p.add_argument("--shade-max", type=float, default=600.0,
                   help="Max shade for uniform bins (ignored when --skew-bins)")
    p.add_argument("--n-bins", type=int, default=42,
                   help="Number of uniform shade bins (ignored when --skew-bins)")
    p.add_argument("--lam-market", type=float, default=None,
                   help="Background market lambda override (default: ENV 0.005)")
    p.add_argument("--shock-var", type=float, default=None,
                   help="Fundamental shock variance override (default: ENV 1e6)")
    p.add_argument("--pv-var", type=float, default=None,
                   help="Private value variance override (default: ENV 5e6)")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: auto-generated from model path)")
    p.add_argument("--legacy", action="store_true",
                   help="Use sinusoidal PE architecture (pre-RoPE checkpoints)")
    args = p.parse_args()

    if args.skew_bins:
        shade_bins = SKEWED_SHADE_BINS
    elif args.shade_max != 600.0 or args.n_bins != 42:
        shade_bins = np.linspace(0, args.shade_max, args.n_bins)
    else:
        shade_bins = UNIFORM_SHADE_BINS

    lam       = args.lam_market if args.lam_market is not None else ENV["lam"]
    shock_var = args.shock_var  if args.shock_var  is not None else ENV["shock_var"]
    pv_var    = args.pv_var     if args.pv_var     is not None else ENV["pv_var"]

    n_procs = args.processes or min(mp.cpu_count(), 8)
    chunk = args.num_runs // n_procs
    remainder = args.num_runs - chunk * n_procs

    rng = np.random.default_rng(args.seed)
    chunks = [chunk + (1 if i < remainder else 0) for i in range(n_procs)]
    seeds = rng.integers(1 << 31, size=n_procs).tolist()

    tasks = [
        (args.model, chunks[i], args.n_bg, args.bg_strategy,
         args.lam_zi, args.n_layers, shade_bins, seeds[i],
         lam, shock_var, pv_var, args.legacy)
        for i in range(n_procs)
    ]

    bin_desc = ("skewed[0,600]x42" if args.skew_bins
                else f"uniform[0,{args.shade_max:.0f}]x{args.n_bins}")
    _sd_peek = torch.load(args.model, map_location="cpu")
    _has_cls  = "cls_token" in _sd_peek
    _has_rope = "rope.inv_freq" in _sd_peek
    if args.legacy or (_has_cls and not _has_rope):
        arch_desc = "legacy(sinusoidal-PE+CLS)"
    elif _has_cls and _has_rope:
        arch_desc = "RoPE+CLS(intermediate)"
    else:
        arch_desc = "current(RoPE,no-CLS)"
    _ffn_hidden = int(_sd_peek["blocks.0.ffn.0.weight"].shape[0]) if "blocks.0.ffn.0.weight" in _sd_peek else 512
    arch_desc += f",ffn={_ffn_hidden}"
    print(f"Running {args.num_runs} episodes across {n_procs} processes")
    print(f"  Model: {args.model}  [{arch_desc}]")
    print(f"  BG: {args.n_bg}x S{args.bg_strategy} ({STRATEGIES[args.bg_strategy]})")
    print(f"  lam={lam}, shock_var={shock_var:.2e}, pv_var={pv_var:.2e}")
    print(f"  lam_zi={args.lam_zi}, n_layers={args.n_layers}, bins={bin_desc}\n")

    all_results = []
    with mp.Pool(n_procs) as pool:
        for chunk_results in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_procs, desc="Chunks"
        ):
            all_results.extend(chunk_results)

    tron_arr = np.array([r[0] for r in all_results])
    bg_arr   = np.array([r[1] for r in all_results])
    n = len(tron_arr)

    def stats(arr):
        m = arr.mean(); s = arr.std(); se = s / np.sqrt(n)
        return m, s, se, np.median(arr), m / se if se > 0 else float('inf')

    tm, ts, tse, tmed, tt = stats(tron_arr)
    bm, bs, bse, bmed, _  = stats(bg_arr)
    rel = (tm - bm) / abs(bm) * 100 if bm != 0 else float('nan')
    diff = tm - bm
    diff_se = np.sqrt(tse**2 + bse**2)
    t_diff = diff / diff_se if diff_se > 0 else float('inf')

    W = 62
    print(f"\n{'='*W}")
    print(f"TRONformer vs {args.n_bg}x S{args.bg_strategy} — {n} episodes")
    print(f"{'='*W}")
    print(f"  {'':30s} {'TRONformer':>12}  {'ZI S8 (mean)':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Mean profit':30s} {tm:>+12.2f}  {bm:>+12.2f}")
    print(f"  {'Std dev':30s} {ts:>12.2f}  {bs:>12.2f}")
    print(f"  {'Std error':30s} {tse:>12.2f}  {bse:>12.2f}")
    print(f"  {'Median':30s} {tmed:>+12.2f}  {bmed:>+12.2f}")
    print(f"  {'t-stat (vs 0)':30s} {tt:>12.2f}")
    print(f"  {'-'*56}")
    print(f"  {'TRONformer advantage':30s} {diff:>+12.2f}  ({rel:>+.1f}%)")
    print(f"  {'t-stat (TRONformer vs ZI S8)':30s} {t_diff:>12.2f}")
    print(f"  {'95% CI (advantage)':30s} [{diff-1.96*diff_se:+.2f}, {diff+1.96*diff_se:+.2f}]")
    print(f"{'='*W}")

    if args.output:
        out = args.output
    else:
        model_dir = os.path.dirname(args.model)
        run_name  = os.path.basename(model_dir)
        out = f"results/equilibrium/{run_name}/dist_{n}runs.csv"

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    df_out = pd.DataFrame({"tron_profit": tron_arr, "bg_mean_profit": bg_arr})
    df_out.to_csv(out, index=False)
    print(f"\nPer-episode profits saved to {out}")


if __name__ == "__main__":
    main()

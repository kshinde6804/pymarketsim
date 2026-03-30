"""
Fresh evaluation: TRONformer (mixed-trained, Env C, seed 42) vs 24 S8 agents.
Reports per-episode profit mean ± std over N_RUNS episodes.
All agents use lam=0.012.
"""

import collections
import random
import sys

import numpy as np
import torch

from marketsim.agent.tronformer_agent import TRONformerPolicy, SEQ_LEN
from marketsim.wrappers.tron_env import TRONEnv

# ── Config ────────────────────────────────────────────────────────────────────

WEIGHTS   = "runs/tronformer_ne24_envC_rope_seed42_skewed/best_model.pt"
N_RUNS    = 500
SEED      = 42
N_LAYERS  = 2
N_BG      = 24
LAM       = 0.012   # background arrival rate
LAM_ZI    = 0.012   # RL agent arrival rate
SHOCK_VAR = 2e4
PV_VAR    = 2e7
REWARD_NORM = 1e3

BG_STRAT = {'shade': [460, 540], 'eta': 0.5}   # S8

SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

NORMALIZERS = {"fundamental": 1e5, "invt": 10, "reward": REWARD_NORM, "pv": 5e5}

# ── Seed ──────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading model: {WEIGHTS}", flush=True)
policy = TRONformerPolicy(input_dim=14, n_layers=N_LAYERS, shade_bins=SKEWED_SHADE_BINS)
policy.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
policy.eval()

# ── Build env ─────────────────────────────────────────────────────────────────

env = TRONEnv(
    num_background_agents=N_BG,
    sim_time=2000,
    lam=LAM,
    lam_zi=LAM_ZI,
    mean=1e5,
    r=0.01,
    shock_var=SHOCK_VAR,
    q_max=10,
    pv_var=PV_VAR,
    shade=[250, 500],
    normalizers=NORMALIZERS,
    bg_strategies=[BG_STRAT],
    warmup_fraction=0.0,
    shade_bins=SKEWED_SHADE_BINS,
)

print(f"Env C: lam={LAM}, lam_zi={LAM_ZI}, shock_var={SHOCK_VAR}, "
      f"pv_var={PV_VAR}, n_bg={N_BG}, sim_time=2000", flush=True)
print(f"Background: 24 × S8 (shade=[460,540], η=0.5)", flush=True)
print(f"Running {N_RUNS} fresh episodes (seed={SEED}) ...\n", flush=True)

# ── Eval loop ─────────────────────────────────────────────────────────────────

dev_profits = []
bg_mean_profits = []   # mean profit across the 24 S8 background agents per episode

for i in range(N_RUNS):
    obs, _ = env.reset()
    obs_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)
    ep_reward = 0.0
    done = False

    while not done:
        obs_buffer.append(obs.astype(np.float32))
        obs_raw = np.stack(list(obs_buffer))               # (cur_len, 14)
        obs_seq = np.zeros((SEQ_LEN, 14), dtype=np.float32)
        obs_seq[SEQ_LEN - len(obs_raw):] = obs_raw         # left-pad
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, 14)

        with torch.no_grad():
            Q_shade, Q_eta = policy(obs_t)                 # (1, n_shade), (1, n_eta)

        shade_idx = int(Q_shade[0].argmax().item())
        eta_idx   = int(Q_eta[0].argmax().item())

        obs, r, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
        ep_reward += r
        done = terminated or truncated

    # Episode ended — capture profits before reset()
    final_fund = env.market.get_final_fundamental()
    bg_ep_profits = [
        env.agents[aid].get_pos_value()
        + env.agents[aid].position * final_fund
        + env.agents[aid].cash
        for aid in range(env.num_agents)
    ]

    profit = ep_reward * REWARD_NORM
    dev_profits.append(profit)
    bg_mean_profits.append(float(np.mean(bg_ep_profits)))

    if (i + 1) % 50 == 0:
        arr = np.array(dev_profits)
        bg_arr = np.array(bg_mean_profits)
        print(f"  episode {i+1:4d}/{N_RUNS}  "
              f"TF mean={arr.mean():8.1f}  BG mean={bg_arr.mean():8.1f}  "
              f"TF std={arr.std():7.1f}  se={arr.std()/np.sqrt(len(arr)):6.1f}",
              flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────

profits   = np.array(dev_profits)
bg_profits = np.array(bg_mean_profits)

tf_mean = profits.mean();    tf_std = profits.std();    tf_se = tf_std / np.sqrt(N_RUNS)
bg_mean = bg_profits.mean(); bg_std = bg_profits.std(); bg_se = bg_std / np.sqrt(N_RUNS)

print(f"\n{'='*60}")
print(f"RESULTS  ({N_RUNS} episodes, TRONformer vs 24×S8, Env C)")
print(f"{'='*60}")
print(f"  TRONformer profit:")
print(f"    Mean          : {tf_mean:8.1f}")
print(f"    Std dev       : {tf_std:8.1f}")
print(f"    Std error     : {tf_se:8.2f}")
print(f"    95% CI        : [{tf_mean - 1.96*tf_se:.1f}, {tf_mean + 1.96*tf_se:.1f}]")
print(f"    Median        : {np.median(profits):8.1f}")
print(f"    Min / Max     : {profits.min():.1f} / {profits.max():.1f}")
print(f"")
print(f"  ZI S8 background agent profit (mean per episode, avg across 24):")
print(f"    Mean          : {bg_mean:8.1f}")
print(f"    Std dev       : {bg_std:8.1f}")
print(f"    Std error     : {bg_se:8.2f}")
print(f"    95% CI        : [{bg_mean - 1.96*bg_se:.1f}, {bg_mean + 1.96*bg_se:.1f}]")
print(f"    Median        : {np.median(bg_profits):8.1f}")
print(f"")
print(f"  TRONformer advantage over avg S8 agent: {tf_mean - bg_mean:+.1f}")
print(f"{'='*60}")

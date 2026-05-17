"""
Evaluate trained TRON models vs 24 S8 background agents in Env C.

Reports per-model: mean profit (raw), ZI S8 baseline profit, and advantage.
Determines whether each model breaks the S8 Nash equilibrium.

Supports paired evaluation (same seed for TRON and ZI deviator per trial),
action-distribution diagnostics, and ensemble Q-value averaging.

Usage:
    python eval_tron_vs_s8_envc.py --num-runs 2000 --paired
    python eval_tron_vs_s8_envc.py --num-runs 500 --diag
    python eval_tron_vs_s8_envc.py --num-runs 2000 --models v3 --ensemble seed42 seed123 seed456
"""
import argparse
import random
import numpy as np
import torch
import multiprocessing as mp
from collections import Counter

from marketsim.agent.tron_agent import TRONPolicy
from marketsim.wrappers.tron_env import TRONEnv
from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent

# Shade bin arrays (must match train_tron.py exactly)
UNIFORM_SHADE_BINS = np.linspace(0, 600, 21)
SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 7)[:-1],
    np.linspace(300, 600, 16),
])
S8_SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 400, 11)[:-1],
    np.linspace(400, 460, 7)[:-1],
    np.linspace(460, 540, 17)[:-1],
    np.linspace(540, 600, 10),
])
FIXED_ETA_BINS = np.array([0.5])

_BINS_BY_COUNT = {
    21: UNIFORM_SHADE_BINS,
    22: SKEWED_SHADE_BINS,
    42: S8_SKEWED_SHADE_BINS,
}

def _load_policy(weights_path: str) -> TRONPolicy:
    """Load TRONPolicy from checkpoint, inferring bins and hidden_dim from weight shapes."""
    ckpt = torch.load(weights_path, map_location='cpu')
    n_shade = ckpt['adv_shade.2.weight'].shape[0]
    n_eta   = ckpt['adv_eta.2.weight'].shape[0]
    shade_bins = _BINS_BY_COUNT.get(n_shade, UNIFORM_SHADE_BINS)
    eta_bins   = FIXED_ETA_BINS if n_eta == 1 else None
    hidden_dim = ckpt['lstm.bias_hh_l0'].shape[0] // 4
    policy = TRONPolicy(input_dim=14, hidden_dim=hidden_dim, shade_bins=shade_bins, eta_bins=eta_bins)
    policy.load_state_dict(ckpt)
    return policy

# ── Env C parameters (must match training) ───────────────────────────────────
ENV_C = {
    'lam':       0.012,
    'lam_zi':    0.012,
    'mean':      1e5,
    'r':         0.01,
    'shock_var': 1e6,
    'pv_var':    2e7,
    'q_max':     10,
    'sim_time':  2000,
    'n_bg':      24,
}

REWARD_NORM = 1e3
PV_NORM = float(np.sqrt(ENV_C['pv_var']))   # ≈ 4472 for pv_var=2e7

S8 = {'shade': [460, 540], 'eta': 0.5}

# Shade bin index closest to S8 midpoint (500) — used as ZI proxy action in paired eval
_S8_SHADE_IDX_42 = int(np.argmin(np.abs(S8_SKEWED_SHADE_BINS - 500)))
_S8_SHADE_IDX_21 = int(np.argmin(np.abs(UNIFORM_SHADE_BINS - 500)))

MODELS = {
    'burnin4_22bin':      'runs/tron_envc_nb24_s8_burnin4/best_model.pt',
    'burnin0_22bin':      'runs/tron_envc_nb24_s8_burnin0/best_model.pt',
    '42bin_burnin0':      'runs/tron_envc_nb24_s8_42bins/best_model.pt',
    '42bin_burnin4':      'runs/tron_envc_nb24_s8_42bins_burnin4/best_model.pt',
    'v2_fixed':           'runs/tron_envc_nb24_s8_v2/best_model.pt',
    'v3':                 'runs/tron_envc_nb24_s8_v3/best_model.pt',
}


# ── ZI S8 baseline (unpaired) ─────────────────────────────────────────────────

def _zi_s8_worker(args):
    num_runs, base_seed = args
    dev_profits = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        sim = Simulator(
            num_background_agents=0,
            sim_time=ENV_C['sim_time'],
            num_assets=1,
            lam=ENV_C['lam'],
            mean=ENV_C['mean'],
            r=ENV_C['r'],
            shock_var=ENV_C['shock_var'],
            q_max=ENV_C['q_max'],
            pv_var=ENV_C['pv_var'],
        )
        sim.agents = {}
        sim.agents[0] = ZIAgent(
            agent_id=0, market=sim.markets[0],
            q_max=ENV_C['q_max'], shade=S8['shade'], eta=S8['eta'],
            pv_var=ENV_C['pv_var'],
        )
        for i in range(1, ENV_C['n_bg'] + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i, market=sim.markets[0],
                q_max=ENV_C['q_max'], shade=S8['shade'], eta=S8['eta'],
                pv_var=ENV_C['pv_var'],
            )

        sim.run()
        fv = sim.markets[0].get_final_fundamental()
        ag = sim.agents[0]
        dev_profits.append(ag.get_pos_value() + ag.position * fv + ag.cash)

    return dev_profits


def run_zi_s8_baseline(num_runs: int, n_proc: int):
    chunk = num_runs // n_proc
    chunks = [chunk] * n_proc
    chunks[-1] += num_runs - chunk * n_proc
    base_seeds = [i * 100_000 for i in range(n_proc)]
    tasks = list(zip(chunks, base_seeds))

    with mp.Pool(n_proc) as pool:
        results = pool.map(_zi_s8_worker, tasks)

    all_profits = [p for sub in results for p in sub]
    return float(np.mean(all_profits)), float(np.std(all_profits, ddof=1) / np.sqrt(len(all_profits)))


# ── Paired eval ───────────────────────────────────────────────────────────────

def _make_tron_env(policy: TRONPolicy):
    """Create a TRONEnv whose bins match a loaded policy."""
    eta_bins = policy.ETA_BINS if len(policy.ETA_BINS) == 1 else None
    return TRONEnv(
        num_background_agents=ENV_C['n_bg'],
        sim_time=ENV_C['sim_time'],
        lam=ENV_C['lam'],
        lam_zi=ENV_C['lam_zi'],
        mean=ENV_C['mean'],
        r=ENV_C['r'],
        shock_var=ENV_C['shock_var'],
        q_max=ENV_C['q_max'],
        pv_var=ENV_C['pv_var'],
        shade=[250, 500],
        normalizers={'fundamental': 1e5, 'invt': 10, 'reward': REWARD_NORM, 'pv': PV_NORM},
        bg_strategies=[{'shade': S8['shade'], 'eta': S8['eta']}],
        shade_bins=policy.SHADE_BINS,
        eta_bins=eta_bins,
    )


def _run_episode_tron(env: TRONEnv, policy: TRONPolicy) -> float:
    """Run one episode with the TRON policy. Returns raw profit (REWARD_NORM already applied)."""
    obs, _ = env.reset()
    h = torch.zeros(1, 1, policy.hidden_dim)
    c = torch.zeros(1, 1, policy.hidden_dim)
    ep_r = 0.0
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Q_s, Q_e, (h, c) = policy(obs_t, (h, c))
        shade_idx = int(Q_s.argmax(dim=-1).item())
        eta_idx   = int(Q_e.argmax(dim=-1).item())
        obs, r, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
        ep_r += r
        done = terminated or truncated
    return ep_r * REWARD_NORM


def _run_episode_zi_proxy(env: TRONEnv) -> float:
    """Run one episode where the RL slot plays a fixed S8-proxy action.

    The ZI proxy always picks the shade bin nearest to 500 (S8 midpoint) and
    eta=0.5 (the middle bin, or index 0 when eta_bins=[0.5]).
    """
    shade_bins = env.SHADE_BINS
    s8_shade_idx = int(np.argmin(np.abs(shade_bins - 500)))
    # eta: middle of ETA_BINS, or index 0 if only one bin
    eta_bins = env.ETA_BINS
    if len(eta_bins) == 1:
        s8_eta_idx = 0
    else:
        s8_eta_idx = int(np.argmin(np.abs(eta_bins - 0.5)))
    fixed_action = np.array([s8_shade_idx, s8_eta_idx])

    obs, _ = env.reset()
    ep_r = 0.0
    done = False
    while not done:
        obs, r, terminated, truncated, _ = env.step(fixed_action)
        ep_r += r
        done = terminated or truncated
    return ep_r * REWARD_NORM


def _paired_worker(args):
    """Run num_runs paired trials. Each trial seeds both TRON and ZI runs identically."""
    policy_path, num_runs, base_seed = args

    policy = _load_policy(policy_path)
    policy.eval()
    env = _make_tron_env(policy)

    advantages = []
    for i in range(num_runs):
        seed = base_seed + i

        # TRON episode
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        tron_profit = _run_episode_tron(env, policy)

        # ZI proxy episode — same seed → same fundamental + PV realisation
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        zi_profit = _run_episode_zi_proxy(env)

        advantages.append(tron_profit - zi_profit)

    return advantages


def run_paired_eval(model_name: str, weights_path: str, num_runs: int, n_proc: int):
    chunk = num_runs // n_proc
    chunks = [chunk] * n_proc
    chunks[-1] += num_runs - chunk * n_proc
    base_seeds = [i * 100_000 for i in range(n_proc)]
    tasks = [(weights_path, c, s) for c, s in zip(chunks, base_seeds)]

    with mp.Pool(n_proc) as pool:
        results = pool.map(_paired_worker, tasks)

    all_adv = [a for sub in results for a in sub]
    mean_adv = float(np.mean(all_adv))
    se_adv   = float(np.std(all_adv, ddof=1) / np.sqrt(len(all_adv)))
    return mean_adv, se_adv


# ── Unpaired TRON eval ────────────────────────────────────────────────────────

def _tron_worker(args):
    model_name, weights_path, num_runs = args

    policy = _load_policy(weights_path)
    policy.eval()
    env = _make_tron_env(policy)

    ep_profits = []
    for _ in range(num_runs):
        ep_profits.append(_run_episode_tron(env, policy))

    return model_name, ep_profits


# ── Ensemble eval ─────────────────────────────────────────────────────────────

def _load_ensemble(paths):
    """Load multiple policies for Q-value averaging."""
    policies = [_load_policy(p) for p in paths]
    for p in policies:
        p.eval()
    return policies


def _run_episode_ensemble(env: TRONEnv, policies) -> float:
    """Run one episode using mean Q-values across an ensemble of policies."""
    obs, _ = env.reset()
    hs = [torch.zeros(1, 1, p.hidden_dim) for p in policies]
    cs = [torch.zeros(1, 1, p.hidden_dim) for p in policies]
    ep_r = 0.0
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        all_Qs = []
        all_Qe = []
        new_hs, new_cs = [], []
        for i, p in enumerate(policies):
            with torch.no_grad():
                Q_s, Q_e, (h2, c2) = p(obs_t, (hs[i], cs[i]))
            all_Qs.append(Q_s)
            all_Qe.append(Q_e)
            new_hs.append(h2)
            new_cs.append(c2)
        hs, cs = new_hs, new_cs
        # Ensemble: average logits then argmax
        mean_Qs = torch.stack(all_Qs).mean(0)
        mean_Qe = torch.stack(all_Qe).mean(0)
        shade_idx = int(mean_Qs.argmax(dim=-1).item())
        eta_idx   = int(mean_Qe.argmax(dim=-1).item())
        obs, r, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
        ep_r += r
        done = terminated or truncated
    return ep_r * REWARD_NORM


# ── Action distribution diagnostic ────────────────────────────────────────────

def run_action_diag(policy: TRONPolicy, n_episodes: int = 200):
    """Log the empirical distribution of shade and eta bin choices."""
    env = _make_tron_env(policy)
    shade_counter: Counter = Counter()
    eta_counter:   Counter = Counter()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        h = torch.zeros(1, 1, policy.hidden_dim)
        c = torch.zeros(1, 1, policy.hidden_dim)
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                Q_s, Q_e, (h, c) = policy(obs_t, (h, c))
            shade_idx = int(Q_s.argmax(dim=-1).item())
            eta_idx   = int(Q_e.argmax(dim=-1).item())
            shade_counter[shade_idx] += 1
            eta_counter[eta_idx]     += 1
            obs, _, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
            done = terminated or truncated

    total = sum(shade_counter.values())
    shade_bins = policy.SHADE_BINS
    eta_bins   = policy.ETA_BINS

    print(f"\n  Action distribution ({n_episodes} episodes, {total} steps):")
    print(f"  Shade bins (top-10 by frequency):")
    for idx, cnt in shade_counter.most_common(10):
        bar = "█" * int(30 * cnt / total)
        print(f"    [{idx:2d}] shade={shade_bins[idx]:6.1f}  {cnt:5d} ({100*cnt/total:5.1f}%)  {bar}")
    print(f"  Eta bins:")
    for idx in sorted(eta_counter.keys()):
        cnt = eta_counter[idx]
        bar = "█" * int(30 * cnt / total)
        print(f"    [{idx:2d}] eta={eta_bins[idx]:.3f}  {cnt:5d} ({100*cnt/total:5.1f}%)  {bar}")

    # Collapse check: is the top-1 shade bin taking >60% of actions?
    top_shade_pct = shade_counter.most_common(1)[0][1] / total
    if top_shade_pct > 0.60:
        top_idx, top_cnt = shade_counter.most_common(1)[0]
        print(f"\n  *** ACTION COLLAPSE DETECTED: shade bin [{top_idx}]={shade_bins[top_idx]:.0f} "
              f"accounts for {100*top_shade_pct:.1f}% of actions ***")
    else:
        print(f"\n  No action collapse detected (top shade bin: {100*top_shade_pct:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-runs', type=int, default=2000)
    p.add_argument('--processes', type=int, default=None)
    p.add_argument('--models', nargs='+', default=None,
                   help='Subset of model keys to evaluate (default: all that exist)')
    p.add_argument('--paired', action='store_true',
                   help='Paired evaluation: same seed for TRON and ZI-proxy per trial')
    p.add_argument('--diag', action='store_true',
                   help='Run action distribution diagnostic on each model')
    p.add_argument('--ensemble', nargs='+', default=None, metavar='PATH',
                   help='Paths to multiple model checkpoints; evaluate as ensemble (avg Q-values)')
    p.add_argument('--ensemble-name', type=str, default='ensemble',
                   help='Label for the ensemble entry in the results table')
    args = p.parse_args()

    n_proc = args.processes or min(mp.cpu_count(), 12)
    num_runs = args.num_runs

    import os
    models = {
        k: v for k, v in MODELS.items()
        if (args.models is None or k in args.models) and os.path.exists(v)
    }

    has_ensemble = bool(args.ensemble)

    print(f"Evaluating TRON model(s) vs ZI S8 baseline in Env C")
    print(f"  lam={ENV_C['lam']}, pv_var={ENV_C['pv_var']:.2e}, n_bg={ENV_C['n_bg']}")
    print(f"  {num_runs} episodes per model  |  {n_proc} processes")
    print(f"  mode: {'paired' if args.paired else 'unpaired'}\n")

    # ── ZI S8 baseline (only needed for unpaired mode) ────────────────────────
    zi_mean, zi_se = None, None
    if not args.paired:
        print("Computing ZI S8 baseline...")
        zi_mean, zi_se = run_zi_s8_baseline(num_runs, n_proc)
        print(f"  ZI S8 baseline: {zi_mean:+.1f} ± {zi_se:.1f}\n")

    # ── Evaluate models ───────────────────────────────────────────────────────
    results = {}   # name → (metric, se, mode)

    for model_name, weights_path in models.items():
        print(f"Evaluating {model_name} ...")

        if args.diag:
            policy = _load_policy(weights_path)
            policy.eval()
            run_action_diag(policy)

        if args.paired:
            mean_adv, se_adv = run_paired_eval(model_name, weights_path, num_runs, n_proc)
            results[model_name] = (mean_adv, se_adv, 'paired')
            broken = "YES" if mean_adv > 2 * se_adv else "no"
            print(f"  {model_name}: paired_advantage={mean_adv:+.1f} ± {se_adv:.1f}  NE broken? {broken}")
        else:
            _, ep_profits = _tron_worker((model_name, weights_path, num_runs))
            mean_p = float(np.mean(ep_profits))
            se_p   = float(np.std(ep_profits, ddof=1) / np.sqrt(len(ep_profits)))
            results[model_name] = (mean_p, se_p, 'unpaired')
            adv = mean_p - zi_mean
            broken = "YES" if adv > 2 * se_p else "no"
            print(f"  {model_name}: profit={mean_p:+.1f} ± {se_p:.1f}  advantage={adv:+.1f}  NE broken? {broken}")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if has_ensemble:
        print(f"\nEvaluating ensemble ({len(args.ensemble)} models): {args.ensemble}")
        policies = _load_ensemble(args.ensemble)
        env = _make_tron_env(policies[0])

        ep_profits = []
        for _ in range(num_runs):
            ep_profits.append(_run_episode_ensemble(env, policies))
        mean_p = float(np.mean(ep_profits))
        se_p   = float(np.std(ep_profits, ddof=1) / np.sqrt(len(ep_profits)))
        ename = args.ensemble_name

        if args.paired:
            print(f"  (Paired eval not implemented for ensemble — showing unpaired profit)")
        adv = mean_p - zi_mean if zi_mean is not None else float('nan')
        broken = "YES" if adv > 2 * se_p else "no"
        results[ename] = (mean_p, se_p, 'ensemble')
        print(f"  {ename}: profit={mean_p:+.1f} ± {se_p:.1f}  advantage={adv:+.1f}  NE broken? {broken}")

    # ── Summary table ─────────────────────────────────────────────────────────
    if not results:
        print("No models evaluated (check --models and model file paths).")
        return

    print("\n" + "=" * 80)
    if args.paired:
        print(f"{'Model':<22}  {'Paired advantage':>18}  {'2×SE threshold':>14}  {'NE broken?':>10}")
        print("-" * 80)
        for name, (val, se, mode) in results.items():
            broken = "YES" if val > 2 * se else "no"
            print(f"{name:<22}  {val:>+14.1f} ±{se:>5.1f}  {2*se:>+14.1f}  {broken:>10}")
        print("=" * 80)
        print("\nPaired advantage = TRON profit − ZI-proxy profit on matched episodes.")
        print("NE broken if paired advantage > 2×SE (95% one-sided).")
    else:
        print(f"{'Model':<22}  {'TRON profit':>14}  {'ZI S8 baseline':>14}  {'Advantage':>10}  {'NE broken?':>10}")
        print("-" * 80)
        for name, (val, se, mode) in results.items():
            adv = val - zi_mean if zi_mean is not None else float('nan')
            broken = "YES" if adv > 2 * se else "no"
            print(f"{name:<22}  {val:>+10.1f} ±{se:>5.1f}  "
                  f"{zi_mean:>+10.1f} ±{zi_se:>5.1f}  {adv:>+10.1f}  {broken:>10}")
        print("=" * 80)
        print(f"\nNE broken if TRON advantage > 2× std-error of TRON profit.")


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()

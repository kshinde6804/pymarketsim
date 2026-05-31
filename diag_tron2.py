"""
diag_tron2.py — Diagnose TRON2 learning failure in Env C.

Two diagnostics to discriminate between two failure hypotheses:

  H1 (bad model selection): Policy learned a genuine advantage, but the 200-run
     raw-profit eval used during training is too noisy (SE≈170) to reliably pick
     the best checkpoint.  Fix: paired advantage model selection (Phase 2A).

  H2 (policy collapse): Policy converged to ZI-mimicry (shade≈500, eta≈0.5),
     never finding a better strategy.  Fix: shadow ZI advantage reward (Phase 2B).

Diagnostic A — Paired advantage eval (--n-paired runs, default 200):
  Run TRON2 and a deterministic S8 ZI policy in the same TRONEnv2 with matching
  random seeds.  Same fundamental + same arrival times + same side assignments →
  market noise cancels.  SE(paired) << SE(independent).

Diagnostic B — Action distribution (--n-dist greedy episodes, default 500):
  Log every (shade_idx, eta_idx) action.  S8 equilibrium maps to (10, 10):
    SHADE_BINS[10] = 500  (mid-range of S8 shade=[460,540])
    ETA_BINS[10]   = 0.5  (S8 eta)
  Concentration at (10,10) → H2 confirmed.

Decision table:
  paired p<0.1, mean_adv>0, actions spread  → H1 → Phase 2A (fix model selection)
  paired p>0.3, actions at/near (10,10)     → H2 → Phase 2B (shadow ZI reward)
  paired p>0.3, actions spread elsewhere    → H3 → investigate reward shaping

Usage:
  python diag_tron2.py --model runs/tron2_envc_s8_v1par/best_model.pt
  python diag_tron2.py --model runs/tron2_envc_s8_v1par/best_model.pt --also-latest
  python diag_tron2.py --model runs/tron2_envc_s8_v1par/best_model.pt --n-paired 500
"""

import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
from scipy import stats

from marketsim.agent.tron2_agent import TRONPolicy2
from marketsim.wrappers.tron2_env import TRONEnv2, SHADE_BINS, ETA_BINS

torch.set_num_threads(1)

HIDDEN_DIM   = 128
ZI_SHADE_IDX = 10   # SHADE_BINS[10] = 500.0 — center of S8 shade=[460,540]
ZI_ETA_IDX   = 10   # ETA_BINS[10]   = 0.5   — S8 eta


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_paired_eval(policy, env_tag, ne_strategy, n_runs, device, seed=0):
    """Paired advantage: TRON2 - ZI, matched seed per episode pair.

    Both runs receive the same fundamental realization, the same BG-agent arrival
    times, and the same RL-agent side assignments.  Market noise cancels in the
    difference, so SE(paired) << SE(independent-comparison).

    Returns
    -------
    advantages : np.ndarray, shape (n_runs,)
        Per-episode (TRON2_profit - ZI_profit).
    tron_steps, zi_steps : lists of int
        Steps per episode for each run (should be equal — sanity check).
    """
    env = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy.eval()
    advantages  = []
    tron_steps  = []
    zi_steps    = []

    for i in range(n_runs):
        # ── TRON2 run ─────────────────────────────────────────────────────────
        _set_seed(seed + i)
        obs, _ = env.reset()
        h_c    = None
        done   = False
        steps  = 0
        while not done:
            obs_t = (torch.tensor(obs, dtype=torch.float32)
                     .unsqueeze(0).unsqueeze(0).to(device))
            with torch.no_grad():
                Q_shade, Q_eta, h_c = policy(obs_t, h_c)
            shade_idx = int(Q_shade.squeeze().argmax().item())
            eta_idx   = int(Q_eta.squeeze().argmax().item())
            obs, _, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated
            steps += 1
        tron_profit = env.get_episode_profit()
        tron_steps.append(steps)

        # ── ZI run — same seed so same fundamental, arrivals, sides ───────────
        _set_seed(seed + i)
        obs, _ = env.reset()
        done   = False
        steps  = 0
        while not done:
            obs, _, terminated, truncated, _ = env.step([ZI_SHADE_IDX, ZI_ETA_IDX])
            done = terminated or truncated
            steps += 1
        zi_profit = env.get_episode_profit()
        zi_steps.append(steps)

        advantages.append(tron_profit - zi_profit)

    policy.train()
    return np.array(advantages), tron_steps, zi_steps


def run_action_distribution(policy, env_tag, ne_strategy, n_runs, device, seed=0):
    """Record (shade_idx, eta_idx) for every action over n_runs greedy episodes.

    Returns
    -------
    action_counts : Counter
        (shade_idx, eta_idx) → total count across all episodes.
    shade_arr, eta_arr : np.ndarray
        All chosen indices (concatenated across episodes).
    """
    env = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy.eval()
    action_counts = Counter()
    shade_list    = []
    eta_list      = []

    for i in range(n_runs):
        _set_seed(seed + 10000 + i)   # offset from paired-eval seeds
        obs, _ = env.reset()
        h_c    = None
        done   = False
        while not done:
            obs_t = (torch.tensor(obs, dtype=torch.float32)
                     .unsqueeze(0).unsqueeze(0).to(device))
            with torch.no_grad():
                Q_shade, Q_eta, h_c = policy(obs_t, h_c)
            shade_idx = int(Q_shade.squeeze().argmax().item())
            eta_idx   = int(Q_eta.squeeze().argmax().item())
            action_counts[(shade_idx, eta_idx)] += 1
            shade_list.append(shade_idx)
            eta_list.append(eta_idx)
            obs, _, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated

    policy.train()
    return action_counts, np.array(shade_list), np.array(eta_list)


def diagnose(model_path, env_tag, ne_strategy, n_paired, n_dist, device, seed=0):
    print(f"\n{'='*65}")
    print(f"Model:  {model_path}")
    print(f"Env:    {env_tag}  |  NE strategy: S{ne_strategy}")
    print(f"{'='*65}")

    if not os.path.exists(model_path):
        print(f"  [SKIP] file not found: {model_path}")
        return None

    policy = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # ── Diagnostic A: Paired advantage ────────────────────────────────────────
    print(f"\n[A] Paired advantage eval  ({n_paired} pairs) …")
    advantages, tron_steps, zi_steps = run_paired_eval(
        policy, env_tag, ne_strategy, n_paired, device, seed
    )

    mean_adv = float(advantages.mean())
    se_adv   = float(advantages.std() / np.sqrt(len(advantages)))
    std_adv  = float(advantages.std())
    _, p_val = stats.ttest_1samp(advantages, 0.0)

    # Sanity: step counts should match (same seed → same arrivals)
    steps_match = all(t == z for t, z in zip(tron_steps, zi_steps))
    steps_msg   = "OK (step counts match)" if steps_match else f"WARNING: mismatch in {sum(t!=z for t,z in zip(tron_steps,zi_steps))} pairs"

    print(f"    Seed sanity (TRON2 steps == ZI steps):  {steps_msg}")
    print(f"    Mean steps/episode: {np.mean(tron_steps):.1f}")
    print()
    print(f"    Advantage (TRON2 − ZI):  {mean_adv:+.2f}  ±  {se_adv:.2f}  (SE)")
    print(f"    Std of paired differences:  {std_adv:.2f}")
    print(f"    t-test vs 0:  p = {p_val:.4f}")
    print(f"    95% CI: [{mean_adv - 1.96*se_adv:.1f}, {mean_adv + 1.96*se_adv:.1f}]")
    print()
    # Compare to what independent comparison would give
    raw_profit_std = 2400   # approximate for Env C
    ind_se = raw_profit_std * np.sqrt(2) / np.sqrt(n_paired)
    print(f"    (For reference: independent comparison SE ≈ {ind_se:.0f}  vs  paired SE = {se_adv:.1f})")

    # ── Diagnostic B: Action distribution ─────────────────────────────────────
    print(f"\n[B] Action distribution  ({n_dist} greedy episodes) …")
    action_counts, shade_arr, eta_arr = run_action_distribution(
        policy, env_tag, ne_strategy, n_dist, device, seed
    )
    total_actions = len(shade_arr)
    s8_count      = action_counts.get((ZI_SHADE_IDX, ZI_ETA_IDX), 0)
    s8_frac       = s8_count / total_actions

    print(f"    Total actions: {total_actions}")
    print(f"    S8 equilibrium (10,10) → shade=500, eta=0.50: "
          f"{s8_count} ({100*s8_frac:.1f}%)")
    print(f"    Unique (shade_idx, eta_idx) pairs: {len(action_counts)}")
    print()

    print("    Top-10 (shade_idx, eta_idx) pairs:")
    for (s, e), cnt in action_counts.most_common(10):
        frac   = cnt / total_actions
        marker = " ← S8" if (s == ZI_SHADE_IDX and e == ZI_ETA_IDX) else ""
        print(f"      ({s:2d},{e:2d})  shade={SHADE_BINS[s]:5.0f}  eta={ETA_BINS[e]:.2f}"
              f"  → {cnt:6d}  ({100*frac:.1f}%){marker}")

    # Shade marginal
    print("\n    Shade marginal (idx  shade  pct  bar):")
    for idx in range(21):
        cnt = int((shade_arr == idx).sum())
        pct = 100 * cnt / total_actions
        bar = "█" * max(0, int(pct * 3))
        marker = " ← S8" if idx == ZI_SHADE_IDX else ""
        print(f"      [{idx:2d}] {SHADE_BINS[idx]:5.0f}  {pct:5.1f}%  {bar}{marker}")

    # Eta marginal
    print("\n    Eta marginal (idx  eta  pct  bar):")
    for idx in range(21):
        cnt = int((eta_arr == idx).sum())
        pct = 100 * cnt / total_actions
        bar = "█" * max(0, int(pct * 3))
        marker = " ← S8" if idx == ZI_ETA_IDX else ""
        print(f"      [{idx:2d}] {ETA_BINS[idx]:.2f}  {pct:5.1f}%  {bar}{marker}")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("INTERPRETATION:")

    # Near-S8 fraction: count (9,*), (10,*), (11,*) shade bins × any eta near 10
    near_s8_shade = int(sum(cnt for (s,e),cnt in action_counts.items() if 8 <= s <= 12))
    near_s8_frac  = near_s8_shade / total_actions

    if p_val < 0.1 and mean_adv > 0:
        verdict = "H1: Policy learned genuine advantage; bad model selection."
        fix     = "Phase 2A — switch training evaluate() to paired advantage."
    elif p_val > 0.3 and near_s8_frac > 0.4:
        verdict = "H2: Policy collapsed to S8-mimicry."
        fix     = "Phase 2B — add shadow ZI advantage reward to TRONEnv2."
    elif p_val > 0.3 and near_s8_frac < 0.2:
        verdict = "H3: Near-zero advantage but actions NOT at S8 — possible reward shaping issue."
        fix     = "Investigate further before choosing a fix."
    else:
        verdict = "Ambiguous — check individual metrics above."
        fix     = "Consider running with more episodes (--n-paired 500 --n-dist 1000)."

    print(f"  {verdict}")
    print(f"  Suggested fix: {fix}")
    print(f"{'─'*65}")

    return {
        "model":       model_path,
        "mean_adv":    mean_adv,
        "se_adv":      se_adv,
        "p_val":       p_val,
        "s8_frac":     s8_frac,
        "near_s8_frac": near_s8_frac,
        "n_unique_actions": len(action_counts),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ZI reference action: shade_idx={ZI_SHADE_IDX} (shade={SHADE_BINS[ZI_SHADE_IDX]:.0f}), "
          f"eta_idx={ZI_ETA_IDX} (eta={ETA_BINS[ZI_ETA_IDX]:.2f})")

    results = [diagnose(
        args.model, args.env_tag, args.ne_strategy,
        args.n_paired, args.n_dist, device, args.seed
    )]

    if args.also_latest:
        latest = os.path.join(os.path.dirname(args.model), "latest_model.pt")
        results.append(diagnose(
            latest, args.env_tag, args.ne_strategy,
            args.n_paired, args.n_dist, device, args.seed
        ))

    # Side-by-side summary if multiple models
    if args.also_latest and all(r is not None for r in results):
        print(f"\n{'='*65}")
        print("SUMMARY")
        print(f"{'='*65}")
        labels = [os.path.basename(args.model), "latest_model.pt"]
        for label, r in zip(labels, results):
            if r:
                print(f"  {label}:")
                print(f"    paired adv = {r['mean_adv']:+.1f} ± {r['se_adv']:.1f}  p={r['p_val']:.3f}"
                      f"  S8-frac={100*r['s8_frac']:.1f}%  unique={r['n_unique_actions']}")


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose TRON2 learning failure")
    p.add_argument("--model",       required=True,
                   help="Path to best_model.pt (or any saved .pt weights)")
    p.add_argument("--env-tag",     default="C", choices=["A", "B", "C"])
    p.add_argument("--ne-strategy", default=8,   type=int)
    p.add_argument("--n-paired",    default=200, type=int,
                   help="Episodes for paired advantage eval (default: 200)")
    p.add_argument("--n-dist",      default=500, type=int,
                   help="Episodes for action distribution (default: 500)")
    p.add_argument("--seed",        default=0,   type=int)
    p.add_argument("--also-latest", action="store_true",
                   help="Also diagnose latest_model.pt in the same run directory")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

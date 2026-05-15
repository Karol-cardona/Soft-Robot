"""
PD Controller Tuning — Grid Search
====================================

Three-stage grid search to find the best (K_P, K_D, signs) for the
classical PD controller. Strategy:

  Stage 1 — Sign detection (4 combos): find correct frame orientation
            with default gains.
  Stage 2 — Gain tuning (4 x 4 grid): with signs fixed, tune (K_P, K_D)
            for best @1cm at the medium training stiffness.
  Stage 3 — Adaptive slopes (4 x 4 grid, ADAPTIVE ONLY): with all the
            above fixed, tune (alpha_P, alpha_D) by validating across
            the full E range and maximizing mean @1cm.

Saves:
  results_classical_pd/fixed_tuned.pth
  results_classical_pd/adaptive_tuned.pth
  results_classical_pd/tuning_log.json

After tuning, validate with validate_student.ipynb / adaptation_test.py
by swapping the loader (see classical_pd_controller.py docstring).

Runtime: ~30-60 min total (most of it in Stage 2 and Stage 3).
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from rod_tracking_env import RodTrackingEnv
from ClassicalMethod.classical_controller import ClassicalPDController

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ======================================================================
# CONFIG
# ======================================================================

OUT_DIR = Path("results_classical_pd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same env params used elsewhere (validate_student, adaptation_test)
ENV_PARAMS = dict(
    n_elem=20,
    sim_dt=2.0e-4,
    num_steps_per_update=7,
    base_length=1.0,
    base_radius=0.05,
    density=1000.0,
    NU=11.0,
    n_control_points=6,
    alpha=75.0,
    max_rate_of_change_of_activation=np.inf,
    target_v_max=0.50,
    p_static=0.0,
    boundary=(-0.35, 0.35, 0.90, 1.0, -0.35, 0.35),
    final_time=10.0,
    success_threshold=0.01,
    w_dist=2.0,
    w_precision=5.0,
    w_progress=1.0,
    w_smoothness=0.03,
    sigma_mult=1.5,
    sigma_floor=0.01,
)

E_MIN, E_MAX = 5e6, 2e7

# Evaluation E used for sign + gain tuning (medium training value)
E_TUNE = 1e7

# E set used for Stage 3 (adaptive slopes); evaluated across the range
E_VAL_ADAPTIVE = [5e6, 1e7, 2e7]   # soft, medium, rigid

# Number of episodes per evaluation — small to keep grid search fast
N_EP_SIGN  = 5     # very few — just to detect correct sign
N_EP_GAIN  = 10    # more for accurate gain tuning
N_EP_ADAPT = 8     # per-E, across 3 E values = 24 episodes total per slope combo


# ======================================================================
# Utilities
# ======================================================================

def normalize_E(E):
    return (np.log10(E) - np.log10(E_MIN)) / (np.log10(E_MAX) - np.log10(E_MIN))


def run_episodes(ctrl, device, E_phys, n_episodes, seed_offset=0):
    """
    Returns dict with @1cm success rate and mean error.
    """
    E_norm = normalize_E(E_phys)
    errors = []
    on_goal_fractions = []
    for ep in range(n_episodes):
        env = RodTrackingEnv(**{**ENV_PARAMS, "young_modulus": E_phys})
        obs, _ = env.reset(seed=ep * 7919 + seed_offset)
        ep_err = []
        done = False
        while not done:
            obs_aug = np.concatenate([obs, [E_norm]]).astype(np.float32)
            with torch.no_grad():
                x = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
                action = torch.tanh(ctrl(x)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ep_err.append(info["error"])
        env.close()
        errors.extend(ep_err)
        on_goal_fractions.append(info["on_goal_fraction"])
    errors = np.array(errors)
    return {
        "success_1cm": float(np.mean(errors < 0.01)) * 100.0,
        "success_2cm": float(np.mean(errors < 0.02)) * 100.0,
        "mean_error":  float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "on_goal":      float(np.mean(on_goal_fractions)),
    }


# ======================================================================
# Stage 1 — Sign detection
# ======================================================================

def stage1_find_signs(device):
    print("\n" + "=" * 70)
    print("  STAGE 1 — Sign detection (4 combinations)")
    print("=" * 70)
    results = []
    for s_n in [+1.0, -1.0]:
        for s_b in [+1.0, -1.0]:
            ctrl = ClassicalPDController(
                K_P=20.0, K_D=5.0,
                sign_normal=s_n, sign_binormal=s_b,
                adaptive=False,
            ).to(device)
            t0 = time.time()
            m = run_episodes(ctrl, device, E_TUNE, N_EP_SIGN)
            dt = time.time() - t0
            print(f"  signs=({s_n:+.0f},{s_b:+.0f})  @1cm={m['success_1cm']:5.1f}%  "
                  f"mean_err={m['mean_error']*100:.2f}cm  ({dt:.1f}s)")
            results.append({
                "sign_normal": s_n, "sign_binormal": s_b, **m
            })
    best = max(results, key=lambda r: r["success_1cm"])
    print(f"\n  → Best signs: ({best['sign_normal']:+.0f}, {best['sign_binormal']:+.0f}) "
          f"with @1cm={best['success_1cm']:.1f}%")
    return best, results


# ======================================================================
# Stage 2 — Gain tuning
# ======================================================================

def stage2_tune_gains(device, sign_normal, sign_binormal):
    print("\n" + "=" * 70)
    print("  STAGE 2 — Gain tuning (K_P x K_D)")
    print("=" * 70)
    K_P_grid = [10.0, 20.0, 40.0, 80.0]
    K_D_grid = [0.0, 2.0, 5.0, 10.0]
    results = []
    n_total = len(K_P_grid) * len(K_D_grid)
    pbar = tqdm(total=n_total, desc="gain grid")
    for K_P in K_P_grid:
        for K_D in K_D_grid:
            ctrl = ClassicalPDController(
                K_P=K_P, K_D=K_D,
                sign_normal=sign_normal, sign_binormal=sign_binormal,
                adaptive=False,
            ).to(device)
            m = run_episodes(ctrl, device, E_TUNE, N_EP_GAIN)
            results.append({"K_P": K_P, "K_D": K_D, **m})
            pbar.set_postfix({"K_P": K_P, "K_D": K_D,
                              "@1cm": f"{m['success_1cm']:.1f}%"})
            pbar.update(1)
    pbar.close()

    print("\n  K_P \\ K_D" + "".join(f"  {kd:>6.1f}" for kd in K_D_grid))
    for K_P in K_P_grid:
        row = f"  {K_P:>6.1f}   "
        for K_D in K_D_grid:
            r = next(x for x in results if x["K_P"] == K_P and x["K_D"] == K_D)
            row += f"  {r['success_1cm']:>5.1f}%"
        print(row)

    best = max(results, key=lambda r: r["success_1cm"])
    print(f"\n  → Best gains: K_P={best['K_P']}, K_D={best['K_D']} "
          f"with @1cm={best['success_1cm']:.1f}%")
    return best, results


# ======================================================================
# Stage 3 — Adaptive slope tuning (only for adaptive controller)
# ======================================================================

def stage3_tune_slopes(device, K_P, K_D, sign_normal, sign_binormal):
    print("\n" + "=" * 70)
    print("  STAGE 3 — Adaptive slope tuning (alpha_P x alpha_D)")
    print("  (evaluated across full E range)")
    print("=" * 70)
    alpha_P_grid = [0.5, 1.0, 2.0, 4.0]
    alpha_D_grid = [0.0, 0.5, 1.0, 2.0]
    results = []
    n_total = len(alpha_P_grid) * len(alpha_D_grid)
    pbar = tqdm(total=n_total, desc="slope grid")
    for aP in alpha_P_grid:
        for aD in alpha_D_grid:
            ctrl = ClassicalPDController(
                K_P=K_P, K_D=K_D,
                sign_normal=sign_normal, sign_binormal=sign_binormal,
                adaptive=True, alpha_P=aP, alpha_D=aD,
            ).to(device)
            # Evaluate across E_VAL_ADAPTIVE
            per_E = {}
            success_list = []
            for E in E_VAL_ADAPTIVE:
                m = run_episodes(ctrl, device, E, N_EP_ADAPT)
                per_E[f"{E:.1e}"] = m
                success_list.append(m["success_1cm"])
            mean_success = float(np.mean(success_list))
            min_success  = float(np.min(success_list))
            results.append({
                "alpha_P": aP, "alpha_D": aD,
                "mean_success_1cm": mean_success,
                "min_success_1cm":  min_success,
                "per_E": per_E,
            })
            pbar.set_postfix({"aP": aP, "aD": aD,
                              "mean@1cm": f"{mean_success:.1f}%"})
            pbar.update(1)
    pbar.close()

    print(f"\n  mean @1cm across {E_VAL_ADAPTIVE}:")
    print("  aP \\ aD" + "".join(f"  {ad:>5.1f}" for ad in alpha_D_grid))
    for aP in alpha_P_grid:
        row = f"  {aP:>5.1f}   "
        for aD in alpha_D_grid:
            r = next(x for x in results
                     if x["alpha_P"] == aP and x["alpha_D"] == aD)
            row += f"  {r['mean_success_1cm']:>5.1f}%"
        print(row)

    # Select on mean success (could also prefer min to penalize worst-case)
    best = max(results, key=lambda r: r["mean_success_1cm"])
    print(f"\n  → Best slopes: alpha_P={best['alpha_P']}, alpha_D={best['alpha_D']} "
          f"with mean@1cm={best['mean_success_1cm']:.1f}%, "
          f"min@1cm={best['min_success_1cm']:.1f}%")
    return best, results


# ======================================================================
# Main
# ======================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Tuning E = {E_TUNE:.1e}  (E_norm = {normalize_E(E_TUNE):.3f})")

    log = {}

    # ---- Stage 1: signs ----
    best_signs, all_signs = stage1_find_signs(device)
    log["stage1_signs"] = {"best": best_signs, "all": all_signs}
    s_n = best_signs["sign_normal"]
    s_b = best_signs["sign_binormal"]

    # ---- Stage 2: gains ----
    best_gains, all_gains = stage2_tune_gains(device, s_n, s_b)
    log["stage2_gains"] = {"best": best_gains, "all": all_gains}
    K_P = best_gains["K_P"]
    K_D = best_gains["K_D"]

    # Save best FIXED controller
    fixed_ctrl = ClassicalPDController(
        K_P=K_P, K_D=K_D,
        sign_normal=s_n, sign_binormal=s_b,
        adaptive=False,
    ).to(device)
    fixed_ctrl.save(OUT_DIR / "fixed_tuned.pth")

    # ---- Stage 3: adaptive slopes ----
    best_slopes, all_slopes = stage3_tune_slopes(device, K_P, K_D, s_n, s_b)
    log["stage3_slopes"] = {"best": best_slopes, "all": all_slopes}

    # Save best ADAPTIVE controller
    adaptive_ctrl = ClassicalPDController(
        K_P=K_P, K_D=K_D,
        sign_normal=s_n, sign_binormal=s_b,
        adaptive=True,
        alpha_P=best_slopes["alpha_P"], alpha_D=best_slopes["alpha_D"],
    ).to(device)
    adaptive_ctrl.save(OUT_DIR / "adaptive_tuned.pth")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"  Best fixed:    K_P={K_P}, K_D={K_D}, "
          f"signs=({s_n:+.0f},{s_b:+.0f})")
    print(f"                 @1cm @ E={E_TUNE:.1e}: "
          f"{best_gains['success_1cm']:.1f}%")
    print(f"  Best adaptive: alpha_P={best_slopes['alpha_P']}, "
          f"alpha_D={best_slopes['alpha_D']}")
    print(f"                 mean @1cm across {E_VAL_ADAPTIVE}: "
          f"{best_slopes['mean_success_1cm']:.1f}%")

    with open(OUT_DIR / "tuning_log.json", "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nLog: {OUT_DIR / 'tuning_log.json'}")
    print(f"Models:")
    print(f"  - {OUT_DIR / 'fixed_tuned.pth'}")
    print(f"  - {OUT_DIR / 'adaptive_tuned.pth'}")

if __name__ == "__main__":
    main()
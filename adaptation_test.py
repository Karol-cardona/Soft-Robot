"""
Adaptation Test — Student Policy
=================================

Tests the student's response to changes in physical body stiffness.
Both the student's E_norm input AND the rod's physical Young's modulus
are synchronized at every step via the env.change_young_modulus() hot-swap
mechanism. This preserves rod state (positions, velocities) while changing
only the elastic response.

Scenarios:
  1. STEP soft → stiff:   E (and E_norm) jump at episode midpoint
  2. STEP stiff → soft:   inverse step
  3. GRADUAL drift:       linear interpolation over the episode

Metrics
-------
For step scenarios:
  - pre_error_cm    : mean error over the 50 steps before the change
                      (initial steady state)
  - peak_error_cm   : max error in the 50 steps after the change
  - recovery_steps  : number of steps to find `RECOVERY_WINDOW`
                      consecutive steps below `RECOVERY_THRESHOLD`
  - post_error_cm   : mean error over the last 50 steps
                      (post-recovery steady state)
  - degradation_factor : peak / pre_mean — how much worse the
                         transient is vs the baseline

For drift scenario:
  - mean error and success-at-1cm in first / middle / last thirds
    of the episode.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from rod_tracking_env import RodTrackingEnv
from training_student import StudentPolicy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# CONFIGURATION

# STUDENT_PATH = "results_dagger_fast/student_dagger_final.pth"
# OUTPUT_DIR   = Path("results_dagger_fast/adaptation_test")

STUDENT_PATH = "results_dagger_targeted/round_1/student_targeted_r1.pth"
OUTPUT_DIR   = Path("results_dagger_targeted/round_1/validation")

E_MIN = 5e6
E_MAX = 2e7

# Physical Young's modulus of the simulated rod during these tests.
E_PHYSICAL = 1e7

ENV_PARAMS = {
    'n_elem': 20,
    'sim_dt': 2.0e-4,
    'num_steps_per_update': 7,
    'base_length': 1.0,
    'base_radius': 0.05,
    'density': 1000.0,
    'NU': 11.0,
    'n_control_points': 6,
    'alpha': 75.0,
    'max_rate_of_change_of_activation': np.inf,
    'target_v_max': 0.50,
    'p_static': 0.0,
    'boundary': (-0.35, 0.35, 0.90, 1.0, -0.35, 0.35),
    'final_time': 10.0,
    'success_threshold': 0.01,
    'w_dist': 2.0,
    'w_precision': 5.0,
    'w_progress': 1.0,
    'w_smoothness': 0.03,
    'sigma_mult': 1.5,
    'sigma_floor': 0.01,
}

N_EPISODES = 30        # replicates per scenario
RECOVERY_THRESHOLD = 0.01    # meters — error level below which we
# consider the agent "recovered"
RECOVERY_WINDOW    = 20

def denormalize_E(E_norm: float) -> float:
    """Inverse of normalize_E: converts E_norm ∈ [0,1] back to E ∈ [E_MIN, E_MAX]."""
    log_E = np.log10(E_MIN) + E_norm * (np.log10(E_MAX) - np.log10(E_MIN))
    return float(10 ** log_E)

# UTILITIES

def normalize_E(E):
    """Log-scale normalize E to [0, 1] — must match training."""
    return (np.log10(E) - np.log10(E_MIN)) / (np.log10(E_MAX) - np.log10(E_MIN))

def load_student(path, device):
    """Load a trained student checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    model = StudentPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Student loaded: ep {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")
    return model


def student_predict(model, obs, E_norm, device):
    """Run the student forward with `obs` augmented by `E_norm`.

    The model outputs pre-tanh logits (per v3 design), so we apply
    tanh here to produce the post-tanh action for env stepping.
    Forgetting this tanh has been a recurring bug in earlier scripts.
    """
    obs_aug = np.concatenate([obs, [E_norm]]).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
        action = torch.tanh(model(x)).squeeze(0).cpu().numpy()
    return action


# SCENARIO RUNNERS

def run_episode_with_E_schedule(model, device, E_initial, E_norm_schedule, seed):
    """
    Run one episode while feeding a per-step E_norm to the student
    AND synchronizing the physical Young's modulus via hot-swap.

    The physical E is updated whenever E_norm changes meaningfully
    (avoids redundant hot-swap calls during constant phases).
    """
    env = RodTrackingEnv(**{**ENV_PARAMS, 'young_modulus': E_initial})
    obs, _ = env.reset(seed=seed)

    errors = []
    E_norms = []
    last_E_phys = E_initial     # track current physical E to detect changes

    step = 0

    # for step in range(N_EPISODES):
    #     E_norm_now = E_norm_schedule(step)
    #
    #     # hot-swap the physical E whenever E_norm changes significantly
    #     E_target = denormalize_E(E_norm_now)
    #     if abs(E_target - last_E_phys) / last_E_phys > 1e-3:  # >0.1% change
    #         env.change_young_modulus(E_target)
    #         last_E_phys = E_target
    #
    #     action = student_predict(model, obs, E_norm_now, device)
    #     obs, _, term, trunc, info = env.step(action)
    #     errors.append(info["error"])
    #     E_norms.append(E_norm_now)
    #     if term or trunc: break

    while True:
        E_norm_now = E_norm_schedule(step)

        # hot-swap the physical E whenever E_norm changes significantly
        E_target = denormalize_E(E_norm_now)
        if abs(E_target - last_E_phys) / last_E_phys > 1e-3:  # >0.1% change
            env.change_young_modulus(E_target)
            last_E_phys = E_target

        action = student_predict(model, obs, E_norm_now, device)
        obs, _, term, trunc, info = env.step(action)

        errors.append(info["error"])
        E_norms.append(E_norm_now)

        if term or trunc:
            break

        step += 1

    env.close()
    return np.array(errors), np.array(E_norms)


def run_step_scenario(model, device, E_norm_before, E_norm_after, n_episodes):
    """Step change in E_norm at the episode midpoint.

    `total_steps` is the per-episode RL-step count (computed once
    upstream so we don't need a per-scenario warm-up).
    """
    all_errors  = []
    all_E_norms = []

    env = RodTrackingEnv(**{**ENV_PARAMS, 'young_modulus': E_PHYSICAL})
    obs, _ = env.reset(seed=0)
    done = False
    total_steps = 0
    while not done:
        action = student_predict(model, obs, E_norm_before, device)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        total_steps += 1
    env.close()
    change_step = total_steps // 2

    for ep in tqdm(range(n_episodes), desc=f"Step {E_norm_before:.2f}→{E_norm_after:.2f}"):
        schedule = lambda step, c=change_step: (
            E_norm_before if step < c else E_norm_after
        )
        errors, E_norms = run_episode_with_E_schedule(
            model, device,  denormalize_E(E_norm_before), schedule, seed=ep * 7919
        )
        all_errors.append(errors)
        all_E_norms.append(E_norms)

    return all_errors, all_E_norms, change_step


def run_drift_scenario(model, device, E_norm_start, E_norm_end, n_episodes):
    """Linear drift of E_norm from `E_norm_start` to `E_norm_end`."""
    all_errors, all_E_norms = [], []

    env = RodTrackingEnv(**{**ENV_PARAMS, 'young_modulus': E_PHYSICAL})
    obs, _ = env.reset(seed=0)
    done = False
    total_steps = 0
    while not done:
        action = student_predict(model, obs, E_norm_start, device)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        total_steps += 1
    env.close()

    schedule = lambda step: E_norm_start + (E_norm_end - E_norm_start) * min(step / total_steps, 1.0)

    for ep in tqdm(range(n_episodes), desc=f"Drift {E_norm_start:.2f}→{E_norm_end:.2f}"):
        errors, E_norms = run_episode_with_E_schedule(
            model, device, denormalize_E(E_norm_start), schedule, seed=ep * 7919
        )
        all_errors.append(errors)
        all_E_norms.append(E_norms)

    return all_errors, all_E_norms


# METRICS

def compute_step_metrics(errors_list, change_step, threshold, window):
    """Per-episode metrics for a step-change scenario.

    Returns
    -------
    list of dicts, one per episode, with:
      pre_error_cm      : mean of last 50 steps before the change
      peak_error_cm     : max in the 50 steps after the change
      post_error_cm     : mean of last 50 steps of the episode
      recovery_steps    : steps until `window` consecutive steps
                          below `threshold` (None if never recovers)
      recovery_time_s   : recovery_steps converted to seconds
      degradation_factor: peak / pre_mean
    """
    metrics = []
    for errors in errors_list:
        if len(errors) < change_step + window:
            continue

        pre_window = errors[max(0, change_step - 50):change_step]
        post_window = errors[change_step:min(change_step + 50, len(errors))]
        end_window = errors[-50:]

        pre_mean = float(np.mean(pre_window))
        peak = float(np.max(post_window)) if len(post_window) > 0 else float('nan')
        end_mean = float(np.mean(end_window))

        # Recovery: first index `i` >= change_step such that errors[i : i+window] are all below threshold.
        recovery = None
        for i in range(change_step, len(errors) - window):
            if np.all(errors[i:i+window] < threshold):
                recovery = i - change_step
                break

        metrics.append({
            "pre_error_cm": pre_mean * 100,
            "peak_error_cm": peak * 100,
            "post_error_cm": end_mean * 100,
            "recovery_steps": recovery,
            "recovery_time_s": recovery * ENV_PARAMS['sim_dt'] * ENV_PARAMS['num_steps_per_update'] if recovery else None,
            "degradation_factor": peak / pre_mean if pre_mean > 1e-6 else float('nan'),
        })

    return metrics


def compute_drift_metrics(errors_list):
    """Per-episode metrics for the drift scenario.

    Splits each episode into 3 equal-length thirds and reports mean
    error (cm) and success-at-1cm (FRACTION in [0,1], not percent) in
    each third. Use the fractions multiplied by 100 if you want
    percent for reports.
    """
    metrics = []
    for errors in errors_list:
        n = len(errors)
        first = errors[:n // 3]
        mid   = errors[n // 3:2 * n // 3]
        last  = errors[2 * n // 3:]

        metrics.append({
            "first_third_mean_cm": float(np.mean(first)) * 100,
            "mid_third_mean_cm":   float(np.mean(mid)) * 100,
            "last_third_mean_cm":  float(np.mean(last)) * 100,
            "first_third_success_1cm": float(np.mean(first < 0.01)),
            "mid_third_success_1cm":   float(np.mean(mid   < 0.01)),
            "last_third_success_1cm":  float(np.mean(last  < 0.01)),
        })
    return metrics


def aggregate(metrics_list, key):
    """Aggregate one metric across episodes, dropping None and NaN.

    Returns mean / std / median / n. When no valid values exist
    returns all-None to signal "couldn't aggregate" downstream.
    """
    vals = [m[key] for m in metrics_list
            if m[key] is not None and not (isinstance(m[key], float) and np.isnan(m[key]))]
    if not vals:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean":   float(np.mean(vals)),
        "std":    float(np.std(vals)),
        "median": float(np.median(vals)),
        "n":      len(vals),
    }


# PLOTTING

def plot_step_scenario(errors_list, E_norms_list, change_step, label, save_path):
    """Step-change diagnostic plot: error trace + E_norm timeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    max_len = max(len(e) for e in errors_list)
    err_matrix = np.full((len(errors_list), max_len), np.nan)
    for i, e in enumerate(errors_list):
        err_matrix[i, :len(e)] = e * 100   # cm

    mean = np.nanmean(err_matrix, axis=0)
    std  = np.nanstd(err_matrix, axis=0)
    p90  = np.nanpercentile(err_matrix, 90, axis=0)
    steps = np.arange(max_len)

    ax1.plot(steps, mean, color='steelblue', lw=1.5, label='Mean')
    ax1.fill_between(steps, mean - std, mean + std, alpha=0.3, color='steelblue', label='±1 σ')
    ax1.plot(steps, p90, color='darkorange', lw=1, alpha=0.7, label='P90')
    ax1.axhline(1.0, color='green', ls='--', alpha=0.5, label='1 cm threshold')
    ax1.axvline(change_step, color='red', ls='--', alpha=0.7, label=f'Step change @ {change_step}')
    ax1.set_ylabel('Error [cm]')
    ax1.set_title(f'Adaptation: {label}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(8, np.nanpercentile(err_matrix, 95) * 1.1)])

    ax2.plot(np.arange(len(E_norms_list[0])), E_norms_list[0],
             color='purple', lw=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('E_norm')
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drift_scenario(errors_list, E_norms_list, label, save_path):
    """Drift-scenario diagnostic plot: error trace + E_norm timeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    max_len = max(len(e) for e in errors_list)
    err_matrix = np.full((len(errors_list), max_len), np.nan)
    for i, e in enumerate(errors_list):
        err_matrix[i, :len(e)] = e * 100

    mean = np.nanmean(err_matrix, axis=0)
    std  = np.nanstd(err_matrix, axis=0)
    p90  = np.nanpercentile(err_matrix, 90, axis=0)
    steps = np.arange(max_len)

    ax1.plot(steps, mean, color='teal', lw=1.5, label='Mean')
    ax1.fill_between(steps, mean - std, mean + std, alpha=0.3, color='teal', label='±1 σ')
    ax1.plot(steps, p90, color='darkorange', lw=1, alpha=0.7, label='P90')
    ax1.axhline(1.0, color='green', ls='--', alpha=0.5, label='1 cm threshold')
    ax1.set_ylabel('Error [cm]')
    ax1.set_title(f'Adaptation: {label}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(8, np.nanpercentile(err_matrix, 95) * 1.1)])

    ax2.plot(np.arange(len(E_norms_list[0])), E_norms_list[0],
             color='purple', lw=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('E_norm')
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = load_student(STUDENT_PATH, device)
    print(f"Physical rod E={E_PHYSICAL:.0e} (E_norm={normalize_E(E_PHYSICAL):.3f})")
    print(f"{N_EPISODES} episodes per scenario\n")

    all_results = {}

    # Scenario 1: STEP soft -> stiff
    print("=" * 60)
    print("  SCENARIO 1: STEP soft -> stiff  (E_norm 0.0 -> 1.0)")
    print("=" * 60)
    errors_1, Es_1, change_1 = run_step_scenario(
        model, device, E_norm_before=0.0, E_norm_after=1.0, n_episodes=N_EPISODES
    )
    metrics_1 = compute_step_metrics(errors_1, change_1, RECOVERY_THRESHOLD, RECOVERY_WINDOW)
    plot_step_scenario(errors_1, Es_1, change_1, "Step soft → stiff",
                       OUTPUT_DIR / "scenario_1_step_soft_to_stiff.png")
    all_results["step_soft_to_stiff"] = {
        "change_step": change_1,
        "pre_error_cm": aggregate(metrics_1, "pre_error_cm"),
        "peak_error_cm": aggregate(metrics_1, "peak_error_cm"),
        "post_error_cm": aggregate(metrics_1, "post_error_cm"),
        "recovery_steps": aggregate(metrics_1, "recovery_steps"),
        "recovery_time_s": aggregate(metrics_1, "recovery_time_s"),
        "degradation_factor": aggregate(metrics_1, "degradation_factor"),
    }

    # Scenario 2: STEP stiff -> soft
    print("\n" + "=" * 60)
    print("  SCENARIO 2: STEP stiff → soft (E_norm 1.0 → 0.0)")
    print("=" * 60)
    errors_2, Es_2, change_2 = run_step_scenario(
        model, device, E_norm_before=1.0, E_norm_after=0.0, n_episodes=N_EPISODES
    )
    metrics_2 = compute_step_metrics(errors_2, change_2, RECOVERY_THRESHOLD, RECOVERY_WINDOW)
    plot_step_scenario(errors_2, Es_2, change_2, "Step stiff → soft",
                       OUTPUT_DIR / "scenario_2_step_stiff_to_soft.png")
    all_results["step_stiff_to_soft"] = {
        "change_step": change_2,
        "pre_error_cm": aggregate(metrics_2, "pre_error_cm"),
        "peak_error_cm": aggregate(metrics_2, "peak_error_cm"),
        "post_error_cm": aggregate(metrics_2, "post_error_cm"),
        "recovery_steps": aggregate(metrics_2, "recovery_steps"),
        "recovery_time_s": aggregate(metrics_2, "recovery_time_s"),
        "degradation_factor": aggregate(metrics_2, "degradation_factor"),
    }

    # Scenario 3: GRADUAL drift
    print("\n" + "=" * 60)
    print("  SCENARIO 3: GRADUAL drift (E_norm 0.0 → 1.0 lineare)")
    print("=" * 60)
    errors_3, Es_3 = run_drift_scenario(
        model, device, E_norm_start=0.0, E_norm_end=1.0, n_episodes=N_EPISODES
    )
    metrics_3 = compute_drift_metrics(errors_3)
    plot_drift_scenario(errors_3, Es_3, "Gradual drift soft → stiff",
                        OUTPUT_DIR / "scenario_3_drift.png")
    all_results["gradual_drift"] = {
        "first_third_mean_cm": aggregate(metrics_3, "first_third_mean_cm"),
        "mid_third_mean_cm":   aggregate(metrics_3, "mid_third_mean_cm"),
        "last_third_mean_cm":  aggregate(metrics_3, "last_third_mean_cm"),
        "first_third_success_1cm": aggregate(metrics_3, "first_third_success_1cm"),
        "mid_third_success_1cm":   aggregate(metrics_3, "mid_third_success_1cm"),
        "last_third_success_1cm":  aggregate(metrics_3, "last_third_success_1cm"),
    }

    # Summary printout
    print("\n" + "=" * 70)
    print("  ADAPTATION SUMMARY")
    print("=" * 70)

    for name, res in all_results.items():
        print(f"\n  {name}:")
        for k, v in res.items():
            if isinstance(v, dict):
                if v.get("mean") is not None:
                    print(f"    {k:<25} mean={v['mean']:.3f} ± {v['std']:.3f} "
                          f"(median={v['median']:.3f}, n={v.get('n', '?')})")
                else:
                    print(f"    {k:<25} no valid values")
            else:
                print(f"    {k:<25} {v}")

    with open(OUTPUT_DIR / "adaptation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
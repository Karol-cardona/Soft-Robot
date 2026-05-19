"""
MRAC vs Student Policy — Hot-Swap Adaptation Comparison
===========================================================

Test scientificamente rigoroso: cambia fisicamente la stiffness
della rod (Young's modulus) a runtime tramite hot-swap delle matrici,
senza resettare le posizioni o interrompere l'integratore.
"""

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from rod_tracking_env import RodTrackingEnv
from training_student import StudentPolicy
from mrac_controller import MRACController

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ======================================================================
# CONFIGURATION
# ======================================================================
STUDENT_PATH = "../results_dagger_targeted/round_1/student_targeted_r1.pth"
OUTPUT_DIR   = Path("results_bonus_hotswap")

SIM_DT               = 2.0e-4
NUM_STEPS_PER_UPDATE = 7
DT_RL                = SIM_DT * NUM_STEPS_PER_UPDATE

N_TOTAL_STEPS   = 7142     # 10 secondi
ACTIVATION_STEP = 3571     # midpoint

N_EPISODES = 30
RECOVERY_THRESHOLD = 0.015  # 1.5 cm
RECOVERY_WINDOW = 20

E_MIN = 5e6
E_MAX = 2e7

ENV_PARAMS_BASE = dict(
    n_elem=20,
    sim_dt=SIM_DT,
    num_steps_per_update=NUM_STEPS_PER_UPDATE,
    base_length=1.0,
    base_radius=0.05,
    density=1000.0,
    NU=11.0,
    n_control_points=6,
    alpha=75.0,
    max_rate_of_change_of_activation=float("inf"),
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

# ======================================================================
# UTILITIES
# ======================================================================
def normalize_E(E: float) -> float:
    return (np.log10(E) - np.log10(E_MIN)) / (np.log10(E_MAX) - np.log10(E_MIN))

def load_student(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StudentPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def student_predict(model, obs: np.ndarray, E_norm: float, device) -> np.ndarray:
    obs_aug = np.concatenate([obs, [E_norm]]).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
        return torch.tanh(model(x)).squeeze(0).cpu().numpy()

def compute_metrics(errors: np.ndarray, activation_step: int) -> dict:
    pre  = errors[:activation_step]
    post = errors[activation_step:]

    pre_mean  = float(np.mean(pre[-50:])) * 100 if len(pre) >= 50 else 0.0
    post_mean = float(np.mean(post[-50:])) * 100 if len(post) >= 50 else 0.0
    peak      = float(np.max(post[:50])) * 100 if len(post) > 0 else 0.0

    recovery = None
    consec = 0
    for i, e in enumerate(post):
        if e < RECOVERY_THRESHOLD:
            consec += 1
            if consec >= RECOVERY_WINDOW:
                recovery = i - RECOVERY_WINDOW + 1
                break
        else:
            consec = 0

    return {
        "pre_error_cm": pre_mean,
        "peak_error_cm": peak,
        "post_error_cm": post_mean,
        "recovery_steps": recovery
    }

def agg(metrics_list: list, key: str, use_median: bool = False) -> dict:
    vals = [m[key] for m in metrics_list if m.get(key) is not None]
    if not vals: return {"mean": None, "median": None, "n": 0}
    if use_median:
        return {"median": float(np.median(vals))}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "n": len(vals),
    }

# ======================================================================
# SCENARIO RUNNER
# ======================================================================
def run_hot_swap_scenario(E1_phys: float, E2_phys: float, student_model, device, mrac: MRACController, n_episodes: int, label: str):
    E1_norm = normalize_E(E1_phys)
    E2_norm = normalize_E(E2_phys)
    student_all, mrac_all = [], []

    for ep in tqdm(range(n_episodes), desc=label):
        seed = ep * 7919

        # ── 1. Student ──
        env = RodTrackingEnv(**{**ENV_PARAMS_BASE, "young_modulus": E1_phys})
        obs, _ = env.reset(seed=seed)
        err_s = []
        for step in range(N_TOTAL_STEPS):
            if step == ACTIVATION_STEP:
                env.change_young_modulus(E2_phys)

            # Lo studente riceve l'informazione aggiornata al volo
            current_E_norm = E1_norm if step < ACTIVATION_STEP else E2_norm

            action = student_predict(student_model, obs, current_E_norm, device)
            obs, _, term, trunc, info = env.step(action)
            err_s.append(info["error"])
            if term or trunc: break
        env.close()
        student_all.append(np.array(err_s))

        # ── 2. MRAC ──
        mrac.reset(keep_gains=False)
        env = RodTrackingEnv(**{**ENV_PARAMS_BASE, "young_modulus": E1_phys})
        obs, _ = env.reset(seed=seed)
        err_m = []
        for step in range(N_TOTAL_STEPS):
            if step == ACTIVATION_STEP:
                env.change_young_modulus(E2_phys)

            action = mrac.predict(obs) # Il MRAC è cieco a E_norm
            obs, _, term, trunc, info = env.step(action)
            err_m.append(info["error"])
            if term or trunc: break
        env.close()
        mrac_all.append(np.array(err_m))

    return student_all, mrac_all

def run_drift_scenario(student_model, device, mrac, n_episodes: int):
    """
    Continuous linear drift of E from E_MIN to E_MAX over the episode.
    Hot-swap performed every step (only when E changes >0.1%) to match
    the student's continuous drift exactly.
    """
    student_all, mrac_all = [], []
    total_steps = N_TOTAL_STEPS  # 7000

    for ep in tqdm(range(n_episodes), desc="Gradual drift"):
        seed = ep * 7919

        # Schedule: linear in log-space (matches student's normalize_E convention)
        def E_at_step(step):
            E_norm = min(step / total_steps, 1.0)
            log_E = np.log10(E_MIN) + E_norm * (np.log10(E_MAX) - np.log10(E_MIN))
            return float(10 ** log_E), E_norm

        # ── 1. Student ──
        env = RodTrackingEnv(**{**ENV_PARAMS_BASE, "young_modulus": E_MIN})
        obs, _ = env.reset(seed=seed)
        err_s = []
        last_E = E_MIN
        for step in range(total_steps):
            E_now, E_norm_now = E_at_step(step)
            if abs(E_now - last_E) / last_E > 1e-3:
                env.change_young_modulus(E_now)
                last_E = E_now
            action = student_predict(student_model, obs, E_norm_now, device)
            obs, _, term, trunc, info = env.step(action)
            err_s.append(info["error"])
            if term or trunc: break
        env.close()
        student_all.append(np.array(err_s))

        # ── 2. MRAC (gains continuously adapt) ──
        mrac.reset(keep_gains=False)
        env = RodTrackingEnv(**{**ENV_PARAMS_BASE, "young_modulus": E_MIN})
        obs, _ = env.reset(seed=seed)
        err_m = []
        last_E = E_MIN
        for step in range(total_steps):
            E_now, _ = E_at_step(step)
            if abs(E_now - last_E) / last_E > 1e-3:
                env.change_young_modulus(E_now)
                last_E = E_now
            action = mrac.predict(obs)
            obs, _, term, trunc, info = env.step(action)
            err_m.append(info["error"])
            if term or trunc: break
        env.close()
        mrac_all.append(np.array(err_m))

    return student_all, mrac_all


def compute_drift_metrics(errors: np.ndarray) -> dict:
    """Per-episode drift metrics split into thirds, matching adaptation_test.py"""
    n = len(errors)
    if n == 0:
        return {}

    first = errors[:n // 3]
    mid   = errors[n // 3:2 * n // 3]
    last  = errors[2 * n // 3:]

    return {
        "first_third_mean_cm": float(np.mean(first)) * 100 if len(first) > 0 else None,
        "mid_third_mean_cm":   float(np.mean(mid)) * 100 if len(mid) > 0 else None,
        "last_third_mean_cm":  float(np.mean(last)) * 100 if len(last) > 0 else None,
        "first_third_success_1cm": float(np.mean(first < 0.01)) if len(first) > 0 else None,
        "mid_third_success_1cm":   float(np.mean(mid < 0.01)) if len(mid) > 0 else None,
        "last_third_success_1cm":  float(np.mean(last < 0.01)) if len(last) > 0 else None,
    }

# ======================================================================
# PLOTTING
# ======================================================================
def plot_comparison(student_errors, mrac_errors, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    for data, color, name in [(student_errors, "steelblue", "Student (RL)"),
                              (mrac_errors, "tomato", "MRAC (Classic)")]:
        M = np.array(data) * 100 # In cm
        mean = np.mean(M, axis=0)
        std = np.std(M, axis=0)
        steps = np.arange(M.shape[1])
        ax.plot(steps, mean, color=color, lw=2, label=f"{name} Mean Error")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.axvline(ACTIVATION_STEP, color="red", ls="--", lw=2, label="Hot-Swap (E changed)")
    ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="1 cm target")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Error [cm]")
    ax.set_xlabel("Steps")
    ax.set_ylim([0, 15])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_drift_comparison(student_errors, mrac_errors, save_path: Path):
    fig, ax = plt.subplots(figsize=(13, 6))

    for data, color, name in [(student_errors, "steelblue", "Student (RL)"),
                              (mrac_errors,    "tomato",    "MRAC (Classic)")]:
        M = np.array(data) * 100
        mean = np.mean(M, axis=0)
        std  = np.std(M, axis=0)
        steps = np.arange(M.shape[1])
        ax.plot(steps, mean, color=color, lw=2, label=f"{name} Mean Error")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    # E annotations at start and end (continuous drift)
    ax.text(0, 14.2, f"E={E_MIN:.1e}", ha="left",  fontsize=10, color="gray")
    ax.text(N_TOTAL_STEPS, 14.2, f"E={E_MAX:.1e}", ha="right", fontsize=10, color="gray")
    ax.text(N_TOTAL_STEPS // 2, 14.2, "← continuous drift in E (log-scale) →",
            ha="center", fontsize=9, color="gray", style="italic")

    ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="1 cm target")
    ax.set_title("Gradual Drift: E = 5e6 → 2e7 Pa (continuous)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Error [cm]")
    ax.set_xlabel("Steps")
    ax.set_ylim([0, 15])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ======================================================================
# MAIN
# ======================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("Loading student policy…")
    student = load_student(STUDENT_PATH, device)

    mrac = MRACController(
        dt=DT_RL,
        n_control_points=6,
        lambda_m=4.0,

        gamma_p=8.0,
        gamma_d=1.5,
        gamma_i=0.3,
        gamma_ff=1.0,

        K_p_init=70.0,
        K_d_init=2.5,
        K_i_init=0.5,
        K_ff_init=3.5,

        gain_bounds=(0.05, 100.0),
        anti_windup=1.0
    )

    print("\n" + "="*60)
    print(" HOT-SWAP ADAPTATION TEST")
    print("="*60)

    # 1. Soft -> Stiff
    s_err_1, m_err_1 = run_hot_swap_scenario(5e6, 2e7, student, device, mrac, N_EPISODES, "Soft -> Stiff")
    met_s_1 = [compute_metrics(e, ACTIVATION_STEP) for e in s_err_1]
    met_m_1 = [compute_metrics(e, ACTIVATION_STEP) for e in m_err_1]
    plot_comparison(s_err_1, m_err_1, OUTPUT_DIR / "hotswap_soft_to_stiff.png", "Hot-Swap: Soft (5e6) to Stiff (2e7)")

    # 2. Stiff -> Soft
    s_err_2, m_err_2 = run_hot_swap_scenario(2e7, 5e6, student, device, mrac, N_EPISODES, "Stiff -> Soft")
    met_s_2 = [compute_metrics(e, ACTIVATION_STEP) for e in s_err_2]
    met_m_2 = [compute_metrics(e, ACTIVATION_STEP) for e in m_err_2]
    plot_comparison(s_err_2, m_err_2, OUTPUT_DIR / "hotswap_stiff_to_soft.png", "Hot-Swap: Stiff (2e7) to Soft (5e6)")

    # 3. Gradual drift
    s_err_3, m_err_3 = run_drift_scenario(student, device, mrac, N_EPISODES)
    plot_drift_comparison(s_err_3, m_err_3, OUTPUT_DIR / "hotswap_gradual_drift.png")

    met_s_3 = [compute_drift_metrics(e) for e in s_err_3]
    met_m_3 = [compute_drift_metrics(e) for e in m_err_3]

    print("\n" + "=" * 80)
    print(f"{'RESULTS':^80}")
    print("=" * 80)

    def print_scenario_res(label, met_s, met_m):
        print(f"\n{label}")
        print("-" * 60)
        print(f"{'Metric':<20} | {'Student':<15} | {'MRAC':<15}")
        print("-" * 60)
        for k in ["pre_error_cm", "peak_error_cm", "post_error_cm", "recovery_steps"]:
            # Use median for recovery_steps (matches adaptation_test.py),
            # mean for error metrics
            agg_key = "median" if k == "recovery_steps" else "mean"
            s_val = agg(met_s, k)[agg_key]
            m_val = agg(met_m, k)[agg_key]
            s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
            m_str = f"{m_val:.2f}" if m_val is not None else "N/A"
            print(f"{k:<20} | {s_str:<15} | {m_str:<15}")

    print_scenario_res("SCENARIO 1: SOFT (5e6) -> STIFF (2e7)", met_s_1, met_m_1)
    print_scenario_res("SCENARIO 2: STIFF (2e7) -> SOFT (5e6)", met_s_2, met_m_2)

    # Drift per-third table (aligned with adaptation_test.py)
    print(f"\nSCENARIO 3: GRADUAL DRIFT (Thirds)")
    print("-" * 65)
    print(f"{'Metric':<25} | {'Student':<15} | {'MRAC':<15}")
    print("-" * 65)

    drift_keys = [
        "first_third_mean_cm", "mid_third_mean_cm", "last_third_mean_cm",
        "first_third_success_1cm", "mid_third_success_1cm", "last_third_success_1cm"
    ]

    for k in drift_keys:
        s_val = agg(met_s_3, k)["mean"]
        m_val = agg(met_m_3, k)["mean"]

        if "success" in k:
            s_str = f"{s_val:.1%}" if s_val is not None else "N/A"
            m_str = f"{m_val:.1%}" if m_val is not None else "N/A"
        else:
            s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
            m_str = f"{m_val:.2f}" if m_val is not None else "N/A"

        print(f"{k:<25} | {s_str:<15} | {m_str:<15}")

if __name__ == "__main__":
    main()
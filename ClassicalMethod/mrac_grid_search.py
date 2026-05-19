"""
MRAC Grid Search — Linear Regime + Sign Exploration
=====================================================

Differenze rispetto alla grid precedente:
  1. K_p testato in [5, 10, 20] invece di [100, 200, 400]
     → opera nel regime lineare di tanh, dove i parametri contano
  2. Include K_d, K_ff (cruciali per tracking dinamico)
  3. Include i segni (SIGN_X, SIGN_Z) come parametri esplorati
  4. Riporta la saturation rate per ogni combinazione
     → se sat_rate > 0.5, ignora quel risultato (sei in bang-bang)
  5. Salva su CSV per analisi successiva

Tempo previsto: ~12-15 minuti per 24 combinazioni × 5 episodi.
"""

import os, csv, itertools
import numpy as np
from tqdm import tqdm

from rod_tracking_env import RodTrackingEnv
from mrac_controller import MRACController

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ======================================================================
# CONFIGURATION
# ======================================================================

SIM_DT = 2.0e-4
NUM_STEPS_PER_UPDATE = 7
DT_RL = SIM_DT * NUM_STEPS_PER_UPDATE
N_TOTAL_STEPS = 5000
ACTIVATION_STEP = 2500
N_EPISODES = 5                   # 5 episodi per stabilità statistica

OUTPUT_CSV = "mrac_grid_results.csv"

ENV_PARAMS_BASE = dict(
    n_elem=20, sim_dt=SIM_DT, num_steps_per_update=NUM_STEPS_PER_UPDATE,
    base_length=1.0, base_radius=0.05, density=1000.0, NU=11.0,
    n_control_points=6, alpha=75.0,
    max_rate_of_change_of_activation=float("inf"),
    target_v_max=0.50, p_static=0.0,
    boundary=(-0.35, 0.35, 0.90, 1.0, -0.35, 0.35),
    final_time=10.0, success_threshold=0.01,
    w_dist=2.0, w_precision=5.0, w_progress=1.0,
    w_smoothness=0.03, sigma_mult=1.5, sigma_floor=0.01,
)

# ── Griglia nel regime LINEARE ──
# K_p * e_tipico * w_max < 1 → K_p < 35 con e=0.10, w_max=0.286
# grid = {
#     "K_p_init": [8.0, 15.0, 25.0],
#     "K_d_init": [1.0, 2.5],
#     "K_ff_init": [0.5, 2.0],
#     "signs":    [(-1.0, +1.0), (+1.0, -1.0)],   # le due combo più probabili
# }

grid = {
    "K_p_init": [35.0, 50.0, 70.0],
    "K_d_init": [2.5],           # fisso al vincitore
    "K_ff_init": [2.0, 3.5, 5.0],     # K_ff=2 ha vinto, prova anche più alto
    "signs":    [(-1.0, +1.0)],  # confermato
}

# ======================================================================
# MRAC con segni custom (subclass)
# ======================================================================

class MRACControllerCustomSigns(MRACController):
    """Variante del MRAC dove SIGN_X, SIGN_Z sono parametri della classe."""
    def __init__(self, sign_x: float, sign_z: float, **kwargs):
        super().__init__(**kwargs)
        self.SIGN_X = sign_x
        self.SIGN_Z = sign_z
        self.action_history = []   # per saturation rate

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Replica integrale del predict() ma con segni iniettati e tracking
        # delle azioni.
        DEADZONE_THRESHOLD = 0.005

        tip_pos    = np.array([obs[10], obs[21], obs[32]])
        tip_vel    = obs[33] * obs[34:37]
        target_pos = obs[37:40]
        target_vel = obs[40] * obs[41:44]

        error      = target_pos - tip_pos
        error_norm = float(np.linalg.norm(error))

        d_error = target_vel - tip_vel
        self._integral_error += error * self.dt
        self._integral_error = np.clip(self._integral_error,
                                       -self.anti_windup, self.anti_windup)

        self._e_ref = self._e_ref * np.exp(-self.lambda_m * self.dt)
        e_model = error - self._e_ref

        if error_norm > DEADZONE_THRESHOLD:
            e_hat  = error / error_norm
            e_proj = float(np.dot(e_model, e_hat))

            phi_p  = error_norm
            phi_d  = float(np.linalg.norm(d_error))
            phi_i  = float(np.linalg.norm(self._integral_error))
            phi_ff = float(np.linalg.norm(target_vel[[0, 2]]))

            self.K_p  += self.gamma_p  * e_proj * phi_p  * self.dt
            self.K_d  += self.gamma_d  * e_proj * phi_d  * self.dt
            self.K_i  += self.gamma_i  * e_proj * phi_i  * self.dt
            self.K_ff += self.gamma_ff * e_proj * phi_ff * self.dt

            self.K_p  = float(np.clip(self.K_p,  self.gain_min, self.gain_max))
            self.K_d  = float(np.clip(self.K_d,  0.0,           self.gain_max))
            self.K_i  = float(np.clip(self.K_i,  0.0,           self.gain_max))
            self.K_ff = float(np.clip(self.K_ff, 0.0,           self.gain_max))
        else:
            self._e_ref = error.copy()

        e_x, e_z   = error[0], error[2]
        de_x, de_z = d_error[0], d_error[2]
        ie_x, ie_z = self._integral_error[0], self._integral_error[2]
        vt_x, vt_z = target_vel[0], target_vel[2]

        u_x = self.SIGN_X * (self.K_p * e_x  + self.K_i * ie_x +
                             self.K_d * de_x + self.K_ff * vt_x) * self._w
        u_z = self.SIGN_Z * (self.K_p * e_z  + self.K_i * ie_z +
                             self.K_d * de_z + self.K_ff * vt_z) * self._w

        action = np.concatenate([u_x, u_z])
        action_clipped = np.clip(action, -self.clip, self.clip)

        self.action_history.append(action_clipped.copy())
        self.gain_history.append((self.K_p, self.K_d, self.K_i, self.K_ff))
        self.error_history.append(error_norm)

        return action_clipped

    def reset(self, keep_gains: bool = False):
        super().reset(keep_gains=keep_gains)
        self.action_history.clear()

    def saturation_rate(self) -> float:
        """Fraction of steps where the action hit the clip bound."""
        if not self.action_history:
            return 0.0
        A = np.array(self.action_history)
        return float(np.mean(np.abs(A) >= 0.99))


# ======================================================================
# RUN ONE COMBINATION
# ======================================================================

def evaluate_params(params: dict, n_episodes: int = N_EPISODES) -> dict:
    """Returns dict with mean post-change error, sat_rate, recovery_steps."""
    sign_x, sign_z = params["signs"]

    post_errors = []
    sat_rates = []

    for ep in range(n_episodes):
        seed = ep * 4567 + 11

        mrac = MRACControllerCustomSigns(
            sign_x=sign_x, sign_z=sign_z,
            dt=DT_RL, n_control_points=6, lambda_m=4.0,
            gamma_p=8.0, gamma_d=1.5, gamma_i=0.3, gamma_ff=1.0,
            K_p_init=params["K_p_init"],
            K_d_init=params["K_d_init"],
            K_i_init=0.5,
            K_ff_init=params["K_ff_init"],
            gain_bounds=(0.05, 100.0),
            anti_windup=1.0,
        )

        env = RodTrackingEnv(**{**ENV_PARAMS_BASE, "young_modulus": 5e6})
        obs, _ = env.reset(seed=seed)

        errs = []
        for step in range(N_TOTAL_STEPS):
            if step == ACTIVATION_STEP:
                env.change_young_modulus(2e7)
            action = mrac.predict(obs)
            obs, _, term, trunc, info = env.step(action)
            errs.append(info["error"])
            if term or trunc:
                break
        env.close()

        # Errore stazionario post-cambiamento (ultimi 200 step in cm)
        post = errs[ACTIVATION_STEP:]
        if len(post) > 200:
            post_errors.append(float(np.mean(post[-200:])) * 100)
        else:
            post_errors.append(100.0)

        sat_rates.append(mrac.saturation_rate())

    return {
        "post_error_cm": float(np.mean(post_errors)),
        "post_error_std": float(np.std(post_errors)),
        "saturation_rate": float(np.mean(sat_rates)),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))
    n_total = len(combinations)

    print(f"\nGrid search: {n_total} combinations × {N_EPISODES} episodes\n")

    results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K_p", "K_d", "K_ff", "sign_x", "sign_z",
                         "post_error_cm", "post_error_std", "saturation_rate"])

        for idx, values in enumerate(combinations):
            params = dict(zip(keys, values))
            sign_x, sign_z = params["signs"]

            res = evaluate_params(params)
            results.append((params, res))

            writer.writerow([
                params["K_p_init"], params["K_d_init"], params["K_ff_init"],
                sign_x, sign_z,
                res["post_error_cm"], res["post_error_std"],
                res["saturation_rate"],
            ])
            f.flush()

            flag = " ⚠ SAT" if res["saturation_rate"] > 0.5 else ""
            print(f"[{idx+1:2d}/{n_total}] "
                  f"K_p={params['K_p_init']:5.1f} K_d={params['K_d_init']:4.1f} "
                  f"K_ff={params['K_ff_init']:4.1f} signs=({sign_x:+.0f},{sign_z:+.0f}) "
                  f"→ err={res['post_error_cm']:5.2f}±{res['post_error_std']:4.2f} cm  "
                  f"sat={res['saturation_rate']:.0%}{flag}")

    # ── Best non-saturated combination ─────────────────────────────────
    valid = [(p, r) for (p, r) in results if r["saturation_rate"] < 0.5]

    print("\n" + "=" * 70)
    if not valid:
        print(" ALL combinations saturated! Lower K_p further (try 2-5).")
    else:
        valid.sort(key=lambda x: x[1]["post_error_cm"])
        best_params, best_res = valid[0]
        print(f" BEST NON-SATURATED — error = {best_res['post_error_cm']:.2f} ± "
              f"{best_res['post_error_std']:.2f} cm")
        print(f" Saturation rate: {best_res['saturation_rate']:.0%}")
        print("=" * 70)
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("\n" + "=" * 70)
        print(f" Top 5 non-saturated configurations:")
        for i, (p, r) in enumerate(valid[:5]):
            sx, sz = p["signs"]
            print(f"  {i+1}. K_p={p['K_p_init']:5.1f} K_d={p['K_d_init']:4.1f} "
                  f"K_ff={p['K_ff_init']:4.1f} signs=({sx:+.0f},{sz:+.0f}) "
                  f"→ {r['post_error_cm']:.2f} cm (sat {r['saturation_rate']:.0%})")

    print(f"\n CSV salvato in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
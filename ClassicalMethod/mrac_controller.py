"""
MIT-rule MRAC with adaptive PID + feed-forward for soft rod tip tracking.

Key tuning principles (fixed wrt previous version):
  * K_p_init = 12 (NOT 200): keeps actions in the linear region of tanh.
    At error = 0.10 m with mean spatial weight 0.17:
        u = 12 * 0.10 * 0.17 ≈ 0.20  (well below saturation at 1.0)
  * Tip-weighted spatial profile: control points near the tip have more
    direct authority on tip position (~4× more than basal points).
  * Adaptation rates increased proportionally to make K_p adapt in
    O(100) steps rather than O(2000).
  * Anti-windup tightened (1.0) since K_i is now also small.
"""

import numpy as np


class MRACController:
    """
    MIT-rule MRAC with adaptive PID + feed-forward.

    Adaptive scalars: K_p, K_d, K_i, K_ff
    Reference model:  ė_ref = -λ_m * e_ref  (exact step)
    MIT update law:   dK/dt = +γ * proj(e_model, ê) * ||input||
    """
    def __init__(
            self,
            dt: float,
            n_control_points: int = 6,
            lambda_m: float = 4.0,
            # Adaptation rates (higher than previous to compensate for
            # the lower starting gains and make adaptation visible in
            # O(100) steps).
            gamma_p: float = 8.0,
            gamma_d: float = 1.5,
            gamma_i: float = 0.3,
            gamma_ff: float = 1.0,
            # Initial gains: low enough to operate in tanh linear region.
            K_p_init: float = 12.0,
            K_d_init: float = 1.5,
            K_i_init: float = 0.5,
            K_ff_init: float = 1.0,
            # Hard bounds on gains — the upper bound prevents adaptation
            # from drifting into the saturation regime.
            gain_bounds: tuple = (0.05, 30.0),
            clip: float = 1.0,
            anti_windup: float = 1.0,
    ):
        self.dt = dt
        self.n_cp = n_control_points
        self.lambda_m = lambda_m

        self.gamma_p  = gamma_p
        self.gamma_d  = gamma_d
        self.gamma_i  = gamma_i
        self.gamma_ff = gamma_ff
        self.clip = clip
        self.gain_min, self.gain_max = gain_bounds
        self.anti_windup = anti_windup

        # ── Initial adaptive gains ──────────────────────────────────────
        self._K_p_init  = float(K_p_init)
        self._K_d_init  = float(K_d_init)
        self._K_i_init  = float(K_i_init)
        self._K_ff_init = float(K_ff_init)

        self.K_p  = self._K_p_init
        self.K_d  = self._K_d_init
        self.K_i  = self._K_i_init
        self.K_ff = self._K_ff_init

        # ── Internal state ──────────────────────────────────────────────
        self._e_ref = np.zeros(3)
        self._integral_error = np.zeros(3)

        # ── Spatial weights: tip-weighted ───────────────────────────────
        # Authority on tip position scales roughly with s₀ * (2L - s₀)/2,
        # which is monotonically increasing for s₀ in [0, L]. Linear weights
        # [1,2,3,4,5,6] are a simple, monotonically increasing approximation.
        profile = np.arange(1, n_control_points + 1, dtype=float)  # [1..6]
        self._w = profile / profile.sum()  # normalized to sum = 1

        # ── Diagnostics ─────────────────────────────────────────────────
        self.gain_history  = []
        self.error_history = []

    # ──────────────────────────────────────────────────────────────────
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Compute 12-D action and update adaptive gains from one obs."""

        # Sign convention from the rod's director frame in PyElastica.
        SIGN_X = -1.0
        SIGN_Z = +1.0

        # Below this error level, freeze adaptation (avoid noise-driven
        # drift when already on target).
        DEADZONE_THRESHOLD = 0.005  # 5 mm

        # ── Extract kinematics ─────────────────────────────────────────
        tip_pos    = np.array([obs[10], obs[21], obs[32]])
        tip_vel    = obs[33] * obs[34:37]
        target_pos = obs[37:40]
        target_vel = obs[40] * obs[41:44]

        error      = target_pos - tip_pos
        error_norm = float(np.linalg.norm(error))

        # ── Derivative & integral updates ──────────────────────────────
        d_error = target_vel - tip_vel
        self._integral_error += error * self.dt
        self._integral_error = np.clip(self._integral_error,
                                       -self.anti_windup, self.anti_windup)

        # ── Reference model: exact exponential step ────────────────────
        self._e_ref = self._e_ref * np.exp(-self.lambda_m * self.dt)
        e_model = error - self._e_ref

        # ── MIT adaptation (skip inside dead zone) ─────────────────────
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
            # Inside dead zone: re-anchor reference model to current error.
            self._e_ref = error.copy()

        # ── Control law (PID + FF) ─────────────────────────────────────
        e_x, e_z   = error[0], error[2]
        de_x, de_z = d_error[0], d_error[2]
        ie_x, ie_z = self._integral_error[0], self._integral_error[2]
        vt_x, vt_z = target_vel[0], target_vel[2]

        u_x = SIGN_X * (self.K_p * e_x  + self.K_i * ie_x +
                        self.K_d * de_x + self.K_ff * vt_x) * self._w
        u_z = SIGN_Z * (self.K_p * e_z  + self.K_i * ie_z +
                        self.K_d * de_z + self.K_ff * vt_z) * self._w

        action = np.concatenate([u_x, u_z])
        action = np.clip(action, -self.clip, self.clip)

        # ── Book-keeping ───────────────────────────────────────────────
        self.gain_history.append((self.K_p, self.K_d, self.K_i, self.K_ff))
        self.error_history.append(error_norm)

        return action

    # ──────────────────────────────────────────────────────────────────
    def reset(self, keep_gains: bool = False):
        """Reset internal state for a new episode."""
        self._e_ref = np.zeros(3)
        self._integral_error = np.zeros(3)
        if not keep_gains:
            self.K_p  = self._K_p_init
            self.K_d  = self._K_d_init
            self.K_i  = self._K_i_init
            self.K_ff = self._K_ff_init
        self.gain_history.clear()
        self.error_history.clear()

    def gain_snapshot(self) -> dict:
        if not self.gain_history:
            return {"K_p": self.K_p, "K_d": self.K_d,
                    "K_i": self.K_i, "K_ff": self.K_ff}
        arr = np.array(self.gain_history)
        return {
            "K_p_mean":  float(arr[:, 0].mean()),
            "K_p_final": float(arr[-1, 0]),
            "K_d_mean":  float(arr[:, 1].mean()),
            "K_i_mean":  float(arr[:, 2].mean()),
            "K_ff_mean": float(arr[:, 3].mean()),
        }

    def __repr__(self) -> str:
        return (f"MRACController(K_p={self.K_p:.2f}, K_d={self.K_d:.2f}, "
                f"K_i={self.K_i:.2f}, K_ff={self.K_ff:.2f})")
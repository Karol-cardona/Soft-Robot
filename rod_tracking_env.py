"""
RodTrackingEnv — Gymnasium environment for 3D target tracking
with a soft Cosserat rod (PyElastica).
=============================================================

DESCRIPTION
-----------
A Gymnasium-compliant environment in which a flexible rod, modelled with
the PyElastica library (Cosserat rod theory), must track a moving target
sphere with its tip in a 3D workspace. The RL controller emits 12 values
that parameterize two muscle-torque B-splines (one in the normal
direction, one in the binormal direction), each with 6 internal control
points distributed along the rod.

The observation follows the convention of the reference paper
(Naughton et al. 2021, "Elastica: A Compliant Mechanics Environment for
Soft Robotic Control", IEEE RA-L): discretized rod positions, tip
velocity, and target kinematic state. Total dimension: 44.

STATIC / DYNAMIC EPISODE PROTOCOL
---------------------------------
Each episode is binary — either fully STATIC (target frozen for the
whole episode) or fully DYNAMIC (target moving at |v| = target_v_max,
changing direction every `direction_change_interval` RL steps). The
decision is made ONCE in `reset()` via a Bernoulli(p_static) draw.
This avoids ambiguity and is equivalent to a discrete domain
randomization over the target regime.

REWARD STRUCTURE
----------------
The total reward combines four components:

    r = r_dist + r_precision + r_progress + r_smooth

    r_dist       = -w_dist * dist^2                       # global gradient
    r_precision  =  w_precision * mixture of Gaussians    # strong local gradient
                    (sigma_far ~ 8 cm + sigma_near ~ 1.5 cm)
    r_progress   =  w_progress * delta(dist)              # rewards approaching
                    (clipped to +/- 5 mm, suppressed during post_turn_cooldown)
    r_smooth     = -w_smoothness * ||a_t - a_{t-1}||^2    # penalises jitter

Historical notes:
    - precision-Gaussian sigma = sigma_mult * success_threshold (default 1.5 cm)
    - sigma_far = 8 * success_threshold (default 8 cm) for global attraction
    - Binary static/dynamic split — no more continuous velocity DR.

OBSERVATION CONTRACT (used by training, validation and PD controller)
---------------------------------------------------------------------
obs in R^44, layout:
    obs[ 0:11]   X coords of 11 discretized rod points (base -> tip)
    obs[11:22]   Y coords of 11 discretized rod points
    obs[22:33]   Z coords of 11 discretized rod points
    obs[33]      rod tip velocity magnitude
    obs[34:37]   rod tip velocity direction (unit vector, 3 comp.)
    obs[37:40]   target sphere position (x, y, z)
    obs[40]      target sphere velocity magnitude
    obs[41:44]   target sphere velocity direction (unit vector)

In particular, the tip is at (obs[10], obs[21], obs[32]).

ACTION CONTRACT
---------------
action in R^12, layout:
    action[0:6]   spline torque amplitudes in the NORMAL direction (d1)
    action[6:12]  spline torque amplitudes in the BINORMAL direction (d2)
All values are clipped to [-1, 1] inside `step`, then scaled by `alpha`
via the `MuscleTorquesWithVaryingBetaSplines` actuator.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check
from elastica.dissipation import AnalyticalLinearDamper

from scipy.interpolate import make_interp_spline
from elastica.external_forces import NoForces
from numba import njit
from collections import defaultdict

# ==========================================================================
# ACTUATOR — spline-distributed muscle torques
# ==========================================================================

class MuscleTorquesWithVaryingBetaSplines(NoForces):
    """
    Apply an internal-torque field distributed along the rod, parameterized
    by a cubic B-spline with vanishing boundary conditions.

    Concept: the RL action specifies the spline values at `n_control_points`
    interior knots equispaced along the rod. At the two endpoints (s = 0
    and s = base_length) the spline is forced to zero (vanishing endpoint
    constraint, as in Naughton et al. 2021). The resulting spline is
    evaluated at the cumulative element positions and multiplied by the
    global scale factor `muscle_torque_scale` (alpha) to obtain the
    per-element torque applied along one local frame direction
    (d1 = normal, d2 = binormal).

    The activation signal is optionally rate-limited so it cannot change
    by more than `max_rate_of_change_of_activation` per call. We pass
    `np.inf` by default (no filtering).

    Key parameters
    --------------
    base_length : float
        Rod rest length [m].
    number_of_control_points : int
        Number of interior spline knots (excludes the two zero-endpoints).
    points_func_array : callable or array
        Source of spline values. If callable, called each simulation step
        with `time`. In practice the env passes a mutable list and rewrites
        it in `step()` (shared-mutable pattern).
    muscle_torque_scale : float
        Global scaling factor (alpha).
    direction : str
        "normal" -> index 0 (d1), "binormal" -> index 1 (d2).
    step_skip : int
        Subsample for the optional torque-profile recorder.
    """

    def __init__(self,
                 base_length,
                 number_of_control_points,
                 points_func_array,
                 muscle_torque_scale,
                 direction,
                 step_skip,
                 max_rate_of_change_of_activation=0.01,
                 **kwargs):
        super().__init__()

        # Map the direction string to an index into `external_torques`.
        if direction == "normal":
            self.direction = 0
        elif direction == "binormal":
            self.direction = 1
        else:
            raise NameError("Direction must be 'normal' or 'binormal'.")

        self.points_array = (
            points_func_array
            if hasattr(points_func_array, "__call__")
            else lambda time_v: points_func_array
        )
        self.base_length = base_length
        self.muscle_torque_scale = muscle_torque_scale
        self.torque_profile_recorder = kwargs.get("torque_profile_recorder", None)
        self.step_skip = step_skip
        self.counter = 0
        self.number_of_control_points = number_of_control_points

        # Knot cache: row 0 = positions along the rod, row 1 = values.
        # The two endpoints (indices 0 and -1) stay at zero by construction.
        self.points_cached = np.zeros((2, number_of_control_points + 2))
        self.points_cached[0, :] = np.linspace(0, base_length, number_of_control_points + 2)
        self.points_cached[1, 1:-1] = np.zeros(number_of_control_points)

        self.max_rate_of_change_of_activation = max_rate_of_change_of_activation
        self.initial_call_flag = 0

    def apply_torques(self, system, time: np.float64 = 0.0):
        # Recompute spline coefficients only if the action actually changed
        # (or on the very first call). This is the hot path during stepping.
        if (not np.array_equal(self.points_cached[1, 1:-1], self.points_array(time))
                or self.initial_call_flag == 0):
            self.initial_call_flag = 1

            # Rate-limited update of the cached knot values.
            self.filter_activation(
                self.points_cached[1, 1:-1],
                np.array(self.points_array(time)),
                self.max_rate_of_change_of_activation
            )

            # Rebuild the spline and pre-compute the per-element torque.
            self.my_spline = make_interp_spline(
                self.points_cached[0], self.points_cached[1]
            )
            cumulative_lengths = np.cumsum(system.lengths)
            self.torque_magnitude_cache = (
                    self.muscle_torque_scale * self.my_spline(cumulative_lengths)
            )

        # Inject the cached torques into the rod's external_torques array
        # along the configured direction (d1 or d2).
        self.compute_muscle_torques(
            self.torque_magnitude_cache,
            self.direction,
            system.external_torques
        )

        if self.counter % self.step_skip == 0 and self.torque_profile_recorder is not None:
            self.torque_profile_recorder["time"].append(time)
            self.torque_profile_recorder["torque_mag"].append(self.torque_magnitude_cache.copy())
            self.torque_profile_recorder["torque"].append(system.external_torques.copy())
            self.torque_profile_recorder["element_position"].append(np.cumsum(system.lengths))
        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(torque_magnitude, direction, external_torques):
        # JIT-compiled additive injection along one component of the local frame.
        for k in range(torque_magnitude.shape[0]):
            external_torques[direction, k] += torque_magnitude[k]

    @staticmethod
    @njit(cache=True)
    def filter_activation(signal, input_signal, max_signal_rate_of_change):
        # In-place rate limiter: signal += sign(diff) * min(rate_cap, |diff|).
        # With rate_cap = inf, this collapses to signal := input_signal.
        signal_difference = input_signal - signal
        signal += np.sign(signal_difference) * np.minimum(
            max_signal_rate_of_change, np.abs(signal_difference)
        )

# ==========================================================================
# Simulator collection
# ==========================================================================

class BaseSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    """
    PyElastica systems collection composed of all the mixins we use:
    constraints (one-end-fixed rod, sphere workspace bounds), connections,
    external forcing (muscle torques), callbacks, and damping.
    """
    pass

# ==========================================================================
# RL ENVIRONMENT
# ==========================================================================

class RodTrackingEnv(gym.Env):
    """
    Cosserat rod tracking a randomly moving target in 3D.

    binary static/dynamic episode split + reward sharpening.
    Observation layout is identical to the Elastica paper
    (Naughton et al. 2021).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 young_modulus=1e7,
                 n_elem=20,
                 sim_dt=2.0e-4,     # internal integrator dt
                 num_steps_per_update=7,
                 base_length=1.0,
                 base_radius=0.05,
                 density=1000.0,
                 NU=11.0,       # linear damping coefficient
                 n_control_points=6,
                 alpha=75.0,        # global torque scale
                 max_rate_of_change_of_activation=np.inf,

                 # Binary static/dynamic mix
                 target_v_max=0.50,        # speed for dynamic episodes [m/s]
                 p_static=0.30,            # fraction of fully-static episodes

                 boundary=(-0.35, 0.35, 0.90, 1.0, -0.35, 0.35),
                 target_position=(-0.4, 0.6, 0.2),
                 final_time=10.0,
                 success_threshold=0.01,
                 dim=3.0,       # legacy field, kept for compat

                 # Reward wights
                 w_dist=2.0,
                 w_precision=5.0,
                 w_progress=1.0,
                 w_smoothness=0.03,
                 sigma_mult=1.5,    # sigma_near = sigma_mult * success_threshold
                 sigma_floor=0.01   # absolute lower bound on sigma_near
                 ):

        super().__init__()

        # Physical parameters
        self.E = float(young_modulus)
        self.n_elem = int(n_elem)
        self.base_length = float(base_length)
        self.base_radius = float(base_radius)
        self.density = float(density)
        self.NU = float(NU)
        self.sim_dt = float(sim_dt)
        self.num_steps_per_update = int(num_steps_per_update)
        self.poisson_ratio = 0.5
        self.n_control_points = int(n_control_points)
        self.alpha = float(alpha)
        self.max_rate_of_change_of_activation = float(max_rate_of_change_of_activation)
        self.dim = float(dim)

        # Target regime
        self.target_v_max = float(target_v_max)
        self.p_static = float(p_static)
        self.boundary = np.array(boundary, dtype=np.float64)
        self.target_position = np.array(target_position, dtype=np.float64)

        # Episode timing
        self.final_time = float(final_time)
        self.h_time_step = self.sim_dt
        # total_steps = number of integrator sub-steps per episode
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        # total_learning_steps = number of RL agent steps per episode
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)

        # Success / reward params
        self.success_threshold = float(success_threshold)
        self.w_dist = float(w_dist)
        self.w_precision = float(w_precision)
        self.w_progress = float(w_progress)
        self.w_smoothness = float(w_smoothness)
        self.sigma_mult = float(sigma_mult)
        self.sigma_floor = float(sigma_floor)

        # Recorder subsampling (for the torque profile recorder).
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Action / observation spaces
        # Action: 2 * n_control_points (normal + binormal spline amplitudes),
        # all in [-1, 1] (clipped in `step`).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.n_control_points,),
            dtype=np.float64
        )

        # Observation: identical layout to the Elastica paper.
        # We sub-sample n_elem+1 rod nodes to ~obs_state_points+1 points.
        # rod positions (~11 pts x 3D) + tip velocity (norm + dir, 4)
        #                              + sphere position (3)
        #                              + sphere velocity (norm + dir, 4)
        # = 3*n_rod_obs + 4 + 3 + 4 = 3*11 + 11 = 44 with the defaults.
        self.obs_state_points = 10
        num_points = max(1, int(self.n_elem / self.obs_state_points))
        self.n_rod_obs = len(np.ones(self.n_elem + 1)[0::num_points])
        obs_dim = self.n_rod_obs * 3 + 11   # = 44
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )

        # Pre-episode internal state
        self.time_tracker = np.float64(0.0)
        self.current_step = 0
        self.on_goal = 0                  # consecutive "on-goal" sim time
        self.on_goal_total = 0.0          # cumulative on-goal sim time
        self.direction_change_interval = 500   # RL steps between random redirections
        self.previous_dist = None         # used by the progress reward
        self.previous_action = None       # used by the smoothness penalty
        self._episode_is_static = False   # set in reset()
        self.post_turn_cooldown = 0       # suppresses r_progress just after a turn
        self.current_episode_v = 0.0      # logged for diagnostics

    # ------------------------------------------------------------------
    # Property: sigma_near for the precision Gaussian
    # ------------------------------------------------------------------

    @property
    def precision_sigma(self):
        return max(self.success_threshold * self.sigma_mult, self.sigma_floor)

    def _apply_velocity(self, v: float):
        """
        Sample a random 3D direction (uniform on the sphere via two angles)
        and apply it as the sphere's velocity with magnitude `v`.

        Note: rand_direction_1 and rand_direction_2 are independent uniforms
        on [0, 2*pi]. The product cos(d1)*sin(d2), sin(d1)*sin(d2), cos(d2)
        is the standard spherical parameterization (not strictly area-uniform,
        but uniform "enough" for redirection events).
        """
        self.rand_direction_1 = np.pi * self.np_random.uniform(0, 2)
        self.rand_direction_2 = np.pi * self.np_random.uniform(0, 2)
        self.sphere.velocity_collection[..., 0] = [
            v * np.cos(self.rand_direction_1) * np.sin(self.rand_direction_2),
            v * np.sin(self.rand_direction_1) * np.sin(self.rand_direction_2),
            v * np.cos(self.rand_direction_2),
            ]

    def _get_obs(self):
        """
       Build the 44-D observation. Order must match the OBSERVATION CONTRACT
       in the module docstring; downstream code (RL student, PD controller,
       validation scripts) depends on this exact layout.
       """
        rod_state = self.shearable_rod.position_collection
        num_points = max(1, int(self.n_elem / self.obs_state_points))

        # Stack X coords, then Y coords, then Z coords of the sub-sampled
        # rod nodes. With n_elem=20 we get 11 points per axis = 33 entries.
        rod_compact_state = np.concatenate((
            rod_state[0][0::num_points],
            rod_state[1][0::num_points],
            rod_state[2][0::num_points],
        ))

        # Tip velocity: magnitude + unit vector (handles zero-velocity safely).
        rod_v = self.shearable_rod.velocity_collection[..., -1]
        rod_v_norm = np.array([np.linalg.norm(rod_v)])
        rod_v_dir = np.divide(rod_v, rod_v_norm,
                              out=np.zeros_like(rod_v), where=rod_v_norm != 0)

        # Sphere position + velocity (magnitude + unit vector).
        sph_pos = self.sphere.position_collection.flatten()
        sph_v = self.sphere.velocity_collection.flatten()
        sph_v_norm = np.array([np.linalg.norm(sph_v)])
        sph_v_dir = np.divide(sph_v, sph_v_norm,
                              out=np.zeros_like(sph_v), where=sph_v_norm != 0)

        return np.concatenate((
            rod_compact_state,
            rod_v_norm, rod_v_dir,
            sph_pos,
            sph_v_norm, sph_v_dir,
        )).astype(np.float64)

    # ------------------------------------------------------------------
    # Runtime setters (used by training callbacks / curriculum / DR)
    # ------------------------------------------------------------------

    def set_p_static(self, p_static: float):
        """Update the static-episode probability at runtime (callback hook)."""
        self.p_static = float(p_static)

    def set_target_velocity(self, new_target_v_max: float):
        """Update the dynamic-episode target speed at runtime (callback hook)."""
        self.target_v_max = float(new_target_v_max)

    def set_success_threshold(self, new_threshold: float):
        """Update the success-distance threshold at runtime (callback hook)."""
        self.success_threshold = float(new_threshold)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        Build a fresh simulator: new rod, new target sphere, new initial
        random pose, and decide once whether the episode is static or
        dynamic. Returns the initial observation and an empty info dict.
        """
        super().reset(seed=seed)

        # Fresh simulator collection and time integrator.
        self.simulator = BaseSimulator()
        self.StatefulStepper = PositionVerlet()

        # Rod aligned with +Y (base at origin), normal along +Z.
        # d1 = normal direction (vertical-Z), d2 = binormal (horizontal-X-ish),
        # d3 = tangent along the rod (Y) at rest.
        start = np.zeros(3)
        direction = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        #  Build the Cosserat rod with Young's modulus E and the standard
        # shear modulus G = E / (2*(1+nu)) assumption.
        self.shearable_rod = CosseratRod.straight_rod(
            self.n_elem, start, direction, normal,
            self.base_length,
            base_radius=self.base_radius,
            density=self.density,
            youngs_modulus=self.E,
            shear_modulus=self.E / (2.0 * (1.0 + self.poisson_ratio))
        )
        self.simulator.append(self.shearable_rod)

        # Analytical linear damping (Rayleigh-type) to absorb high-frequency modes.
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=self.NU,
            time_step=self.sim_dt
        )

        # Target spawn: rejection sampling within reachable radius
        tip_rest_pos = np.array([0.0, 1.0, 0.0])
        while True:
            t_pos = np.array([
                self.np_random.uniform(self.boundary[i * 2], self.boundary[i * 2 + 1])
                for i in range(3)
            ])
            if np.linalg.norm(t_pos - tip_rest_pos) <= 0.35:
                break

        # Build the target sphere.
        self.sphere = Sphere(center=t_pos, base_radius=0.05, density=1000)
        self.trajectory_iteration = 0

        # Align the sphere's director frame with the rod's
        self.sphere.director_collection[..., 0] = (
            self.shearable_rod.director_collection[..., 0]
        )

        # Binary static/dynamic decision (made once, here only)
        if self.np_random.random() < self.p_static:
            self.sphere.velocity_collection[..., 0] = [0.0, 0.0, 0.0]
            self._episode_is_static = True
            self.current_episode_v = 0.0
        else:
            self.current_episode_v = self.target_v_max
            self._apply_velocity(self.target_v_max)
            self._episode_is_static = False

        self.simulator.append(self.sphere)

        # Workspace boundary for the target sphere
        # The target must stay within the rod's reachable region: vertical
        # band [y_min, y_max] AND inside a sphere of radius `max_radius`
        # around the workspace center. Hitting either boundary causes a
        # specular-style velocity reflection (so the target keeps moving
        # but bounces off the bounds).
        class WorkspaceBoundaryForSphere(FreeRod):
            def __init__(self, y_min, y_max, center, max_radius, **kwargs):
                super().__init__(**kwargs)
                self.y_min = y_min
                self.y_max = y_max
                self.center = np.array(center)
                self.max_radius = max_radius

            def constrain_values(self, sphere, time):
                p = sphere.position_collection[:, 0]
                r = sphere.radius
                v = sphere.velocity_collection[:, 0]

                # Vertical bounce on the bottom plate.
                if (p[1] - r) < self.y_min:
                    sphere.velocity_collection[1, 0] = np.abs(v[1])
                # Vertical bounce on the top plate.
                elif (p[1] + r) > self.y_max:
                    sphere.velocity_collection[1, 0] = -np.abs(v[1])

                # Spherical-shell bounce: specular reflection if the sphere
                # crosses the outer radius while moving outward.
                rel_p = p - self.center
                dist = np.linalg.norm(rel_p)
                if dist + r > self.max_radius:
                    n = rel_p / dist
                    v_dot_n = np.dot(v, n)
                    if v_dot_n > 0:
                        sphere.velocity_collection[:, 0] = v - 2 * v_dot_n * n

            def constrain_rates(self, sphere, time):
                # Velocity-only constraint; no acceleration intervention.
                pass

        self.simulator.constrain(self.sphere).using(
            WorkspaceBoundaryForSphere,
            y_min=self.boundary[2],
            y_max=self.boundary[3],
            center=[0.0, 1.0, 0.0],
            max_radius=0.35
        )
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,)
        )

        # Attach the two muscle-torque actuators
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.n_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction="normal",
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.n_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction="binormal",
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir
        )

        # Finalize the simulator and extract the stepper interface.
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # Reset per-episode counters / state.
        self.on_goal = 0
        self.on_goal_total = 0.0
        self.current_step = 0
        self.time_tracker = np.float64(0.0)
        self.previous_action = None
        self.previous_dist = None

        return self._get_obs(), {}

    def step(self, action):
        """
        Apply one RL action (= `num_steps_per_update` integrator sub-steps),
        compute the reward, and return the standard Gymnasium 5-tuple.
        """
        # Clamp the action into [-1, 1] and load it into the two spline buffers shared with the actuators.
        action = np.clip(action, -1.0, 1.0)
        self.spline_points_func_array_normal_dir[:] = action[:self.n_control_points]
        self.spline_points_func_array_binormal_dir[:] = action[self.n_control_points:]

        # Integrate physics for `num_steps_per_update` sub-steps.
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper, self.stages_and_updates,
                self.simulator, self.time_tracker, self.time_step
            )

        # Numerical-blow-up guard: NaNs in positions => terminate hard with a
        # heavy penalty. Returns terminated=True so the policy learns to
        # avoid catastrophic action sequences.
        if _isnan_check(self.shearable_rod.position_collection):
            return self._get_obs(), -100.0, True, False, {
                "error": 10.0, "is_success": False, "crash": True,
                "is_success_1cm": False, "is_success_15mm": False,
                "is_success_2cm": False, "is_success_5mm": False,
                "episode_v": self.current_episode_v,
            }

        # Target redirection
        # On dynamic episodes, re-randomize the target direction every
        # `direction_change_interval` RL steps. Speed stays at target_v_max.
        # The redirection sets `post_turn_cooldown` to 60 to suppress the
        # progress reward briefly (otherwise spurious negative deltas right
        # after a turn would be penalized unfairly).
        self.trajectory_iteration += 1
        if self.trajectory_iteration == self.direction_change_interval:
            if not self._episode_is_static:
                self._apply_velocity(self.target_v_max)
            self.trajectory_iteration = 0
            self.post_turn_cooldown = 60

        self.current_step += 1
        state = self._get_obs()

        # Tip-to-target distance.
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )

        # ==================== REWARD COMPONENTS ====================

        # (A) Quadratic distance — provides a globally smooth gradient that
        # always pulls the tip toward the target.
        r_dist = -self.w_dist * (dist ** 2)

        # (B) Precision Gaussian — mixture of a wide (sigma_far ~ 8 cm) and
        # a narrow (sigma_near ~ 1.5 cm) bell. The wide bell gives a soft
        # global attraction; the narrow bell provides the strong gradient
        # in the final centimetre that drives precise on-target behaviour.
        sigma_near = self.precision_sigma
        sigma_far  = max(self.success_threshold * 8.0, 0.10)   # 8 cm minimum

        r_precision = self.w_precision * (
                0.3 * np.exp(-(dist**2) / (2.0 * sigma_far**2))
                + 0.7 * np.exp(-(dist**2) / (2.0 * sigma_near**2))
        )

        # (C) Progress reward — rewards every step that reduces distance.
        # Clipped to +/- 5 mm per step (so a single huge swing doesn't
        # dominate). Disabled during `post_turn_cooldown` to avoid
        # penalizing the agent for the discontinuity introduced by a
        # target redirection.
        r_progress = 0.0
        if self.previous_dist is not None and self.post_turn_cooldown == 0:
            delta = self.previous_dist - dist
            r_progress = self.w_progress * np.clip(delta, -0.005, 0.005)

        # (D) Smoothness penalty — quadratic on the action delta. Active at
        # all times; weight is small so it does not overpower the tracking
        # signal.
        r_smooth = 0.0
        if self.previous_action is not None:
            r_smooth = -self.w_smoothness * np.sum(
                (action - self.previous_action) ** 2
            )

        reward = r_dist + r_precision + r_progress + r_smooth

        # ==================== SUCCESS TRACKING ====================
        is_success = dist < self.success_threshold
        # Each RL step covers `num_steps_per_update` sim sub-steps.
        dt_per_rl_step = self.num_steps_per_update * self.time_step
        if is_success:
            self.on_goal += self.time_step
            self.on_goal_total += dt_per_rl_step
        else:
            # `on_goal` counts CONSECUTIVE on-goal time and resets on miss;
            # `on_goal_total` is cumulative and does not reset.
            self.on_goal = 0

        truncated = self.current_step >= self.total_learning_steps

        self.previous_action = action.copy()
        self.previous_dist = float(dist)

        # Aggregate diagnostics.
        on_goal_fraction = self.on_goal_total / self.final_time
        tip_speed = float(np.linalg.norm(self.shearable_rod.velocity_collection[..., -1]))

        info = {
            "ctime": float(self.time_tracker),
            "error": float(dist),
            "on_goal": float(self.on_goal),
            "on_goal_fraction": float(on_goal_fraction),
            "is_success": is_success,
            "is_success_1cm": bool(dist < 0.01),
            "is_success_15mm": bool(dist < 0.015),
            "is_success_2cm": bool(dist < 0.02),
            "is_success_5mm": bool(dist < 0.005),
            "tip_speed": tip_speed,
            "episode_v": self.current_episode_v,
            "crash": False,
            "distance_reward": float(r_dist),
            "precision_reward": float(r_precision),
            "progress_reward": float(r_progress),
            "smoothness_penalty": float(r_smooth),
            "tip_pos": self.shearable_rod.position_collection[..., -1].tolist(),
            "target_pos": self.sphere.position_collection[..., 0].flatten().tolist(),
        }

        # Decrement the post-turn cooldown counter (used by r_progress).
        if self.post_turn_cooldown > 0:
            self.post_turn_cooldown -= 1

        return state, float(reward), False, truncated, info


    def render(self, mode="human"):
        pass

    @property
    def rod(self):
        """Alias for the rod object."""
        return self.shearable_rod

    @property
    def target_pos(self):
        """Current 3D position of the target sphere (xyz array)."""
        return self.sphere.position_collection[..., 0].flatten()

    @property
    def target_vel(self):
        """Current 3D velocity of the target sphere (xyz array)."""
        return self.sphere.velocity_collection[..., 0].flatten()
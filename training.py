"""
SAC Expert Training for the Soft Rod Tracking Task
====================================================

Trains ONE SAC expert at a fixed Young's modulus (rod stiffness). The
expert learns to track a moving sphere target with the tip of a Cosserat
rod fixed at the base.

This script is run multiple times — once per stiffness — to produce the
set of teacher policies used in the policy-distillation step:

    young_modulus | out_dir                       | role
    ---------------------------------------------------------------------
    5e6 Pa        | results_expert_E5.0e6/        | soft expert
    7.5e6 Pa      | results_expert_E7.5e6/        | medium-soft expert
    1e7 Pa        | results_expert_E1.0e7/        | medium expert
    2e7 Pa        | results_expert_E2.0e7/        | rigid expert

To train a different expert, edit `young_modulus` and `out_dir` in main().

Key training choices:
  * Domain randomization on target motion: 70% dynamic episodes
    (target moving at v_max=0.5 m/s) and 30% static (target stationary).
    This avoids the static-vs-dynamic asymmetry observed when training
    on dynamic only.
  * Best model selection uses on_goal_fraction on DYNAMIC episodes only.
    Static episodes can inflate on_goal because the target never moves,
    so filtering on dynamic gives a stricter, more honest signal.
  * Observations and rewards are normalized via VecNormalize; the
    statistics MUST be saved alongside the model for inference.
"""

import time, json
import numpy as np
from pathlib import Path
from collections import deque

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from rod_tracking_env import RodTrackingEnv

# ======================================================================
# Logging callback — tracks training metrics and saves the best model
# ======================================================================

class LoggingCallback(BaseCallback):
    """
    Tracks per-step and per-episode metrics across all vectorized envs and
    periodically saves the best model.

    Best-model selection metric
    ---------------------------
    We use on_goal_fraction restricted to DYNAMIC episodes (episode_v > 0.1
    m/s). The env mixes static (p_static=0.30) and dynamic episodes; static
    on_goal is artificially high because the target does not move, so a
    global average would reward "doing nothing well" rather than tracking
    a moving target. Filtering on dynamic episodes gives the harder, more
    representative signal we actually care about.

    Logging cadence
    ---------------
    Two independent cadences are used:
      * `log_every_episodes`: prints aggregate metrics every N completed
        episodes (rolling means over `eval_window` episodes).
      * `log_every_steps`: prints mean error every N env steps for a
        faster heartbeat between episode summaries.
    """

    def __init__(self, out_dir, eval_window=200,
                 log_every_episodes=50, log_every_steps=10_000, verbose=1):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.eval_window = eval_window
        self.log_every_episodes = log_every_episodes
        self.log_every_steps = log_every_steps

        # Rolling windows over completed episodes — used for episode-level
        # aggregates (mean reward, mean error, on_goal fraction).
        self.success_window  = deque(maxlen=eval_window)
        self.error_window    = deque(maxlen=eval_window)
        self.reward_window   = deque(maxlen=eval_window)

        # Rolling per-STEP success flags at different thresholds, useful
        # to see how the policy is doing right now (vs episode aggregates
        # which lag behind).
        self.success_1cm_steps  = deque(maxlen=1000)
        self.success_15mm_steps = deque(maxlen=1000)
        self.success_2cm_steps  = deque(maxlen=1000)
        self.success_5mm_steps  = deque(maxlen=1000)

        # Dynamic-only versions (filtered to episodes with v > 0.1 m/s).
        self.dynamic_1cm_steps = deque(maxlen=1000)
        self.dynamic_on_goal = deque(maxlen=200)

        self.episode_count      = 0
        self.last_logged_episode = -1
        self.last_dist_print    = 0
        self.best_success_rate  = -1.0

    def _on_step(self):
        # Each step collects info dicts from every vectorized env. Some
        # keys appear every step (per-step success flags), others only
        # at episode boundaries ("episode", "on_goal_fraction", ...).
        for info in self.locals.get("infos", []):

            # Per-step success flags at various distance thresholds.
            if "is_success_1cm"  in info:
                self.success_1cm_steps.append(float(info["is_success_1cm"]))
                # Track the dynamic-only version separately.
                if info.get("episode_v", 0) > 0.1:
                    self.dynamic_1cm_steps.append(float(info["is_success_1cm"]))
            if "is_success_15mm" in info:
                self.success_15mm_steps.append(float(info["is_success_15mm"]))
            if "is_success_2cm"  in info:
                self.success_2cm_steps.append(float(info["is_success_2cm"]))
            if "is_success_5mm"  in info:
                self.success_5mm_steps.append(float(info["is_success_5mm"]))

            # End-of-episode payload (SB3 wraps the env in VecMonitor which
            # injects the "episode" key on done).
            if "episode" in info:
                self.episode_count += 1
                self.reward_window.append(info["episode"]["r"])

                if "on_goal_fraction" in info:
                    self.success_window.append(float(info["on_goal_fraction"]))
                    if info.get("episode_v", 0) > 0.1:
                        self.dynamic_on_goal.append(float(info["on_goal_fraction"]))

                if "error" in info:
                    self.error_window.append(info["error"])

        # Episode-level log (every N episodes)
        if (self.episode_count > 0
                and self.episode_count % self.log_every_episodes == 0
                and self.episode_count != self.last_logged_episode):

            mr  = np.mean(self.reward_window)        if self.reward_window        else 0.0
            me  = np.mean(self.error_window)          if self.error_window          else -1.0
            sr  = np.mean(self.success_window)        if self.success_window        else 0.0
            s1  = np.mean(self.success_1cm_steps)     if self.success_1cm_steps     else 0.0
            s15 = np.mean(self.success_15mm_steps)    if self.success_15mm_steps    else 0.0
            s2  = np.mean(self.success_2cm_steps)     if self.success_2cm_steps     else 0.0
            s5  = np.mean(self.success_5mm_steps)     if self.success_5mm_steps     else 0.0
            dg = np.mean(self.dynamic_on_goal) if self.dynamic_on_goal else 0.0

            print(f"[Ep {self.episode_count:>6d} | Step {self.num_timesteps:,}] "
                  f"R: {mr:>9.1f} | Err: {me:.4f} m | "
                  f"EpSucc: {sr:.3f} | OnGoal(dyn): {dg:.3f} | "
                  f"@2cm: {s2:.3f} @15mm: {s15:.3f} @1cm: {s1:.3f} @5mm: {s5:.3f}")
            self.last_logged_episode = self.episode_count

        # Step-level heartbeat (mean distance every N steps)
        if (self.num_timesteps - self.last_dist_print) >= self.log_every_steps:
            errs = [i.get("error") for i in self.locals.get("infos", []) if "error" in i]
            if errs:
                print(f"   ---> [Step {self.num_timesteps:,}] Dist: {np.mean(errs):.4f} m")
            self.last_dist_print = self.num_timesteps

        # Try to save the best model (no-op if buffer is too small).
        self._save_best_model()
        return True

    def _save_best_model(self):
        """
        Save the model whenever the rolling mean of on_goal_fraction on
        DYNAMIC episodes exceeds the current best.

        Requires at least 50 dynamic episodes in the buffer to avoid noisy
        early-training swings.

        Both the SAC weights AND the VecNormalize statistics (running
        mean/std of obs and reward) are persisted: inference requires
        both, otherwise the obs normalization would be wrong.
        """
        if len(self.dynamic_on_goal) < 50:
            return
        cr = np.mean(self.dynamic_on_goal)
        if cr > self.best_success_rate:
            self.best_success_rate = cr
            self.model.save(str(self.out_dir / "best_model"))
            self.training_env.save(str(self.out_dir / "vecnorm_best.pkl"))
            print(f"   >>> Best model saved! on_goal(dynamic)={cr:.3f} "
                  f"(Step {self.num_timesteps:,})")


    def _on_training_end(self):
        """Dump a final JSON summary of the run."""
        summary = {
            "total_timesteps":   self.num_timesteps,
            "total_episodes":    self.episode_count,
            "final_mean_error":  float(np.mean(self.error_window)   if self.error_window   else -1),
            "final_on_goal_frac": float(np.mean(self.success_window) if self.success_window else 0),
            "final_1cm_rate":    float(np.mean(self.success_1cm_steps)  if self.success_1cm_steps  else 0),
            "final_2cm_rate":    float(np.mean(self.success_2cm_steps)  if self.success_2cm_steps  else 0),
            "best_on_goal_dynamic": self.best_success_rate,
        }
        with open(self.out_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining summary saved.")
        print(f"  Final error:      {summary['final_mean_error']:.4f} m")
        print(f"  Final on_goal:    {summary['final_on_goal_frac']:.3f}")
        print(f"  Final @1cm:       {summary['final_1cm_rate']:.3f}")
        print(f"  Best @1cm (train):{summary['best_on_goal_dynamic']:.3f}")

# ======================================================================
# Env factory for SubprocVecEnv
# ======================================================================

def make_env(rank, seed, env_params):
    """
    Returns a thunk (no-arg callable) that constructs one RodTrackingEnv.

    SubprocVecEnv spawns one subprocess per env and calls this thunk
    inside the child to instantiate the env there (the simulator is
    not picklable across processes). Each worker is seeded uniquely to
    diversify experience collected in parallel.
    """
    def _init():
        env = RodTrackingEnv(**env_params)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    total_steps = 15_000_000
    n_envs = 16
    seed = 42

    env_params = {
        'n_elem': 20,
        'sim_dt': 2.0e-4,
        'num_steps_per_update': 7,
        'base_length': 1.0,
        'base_radius': 0.05,
        'density': 1000.0,
        'young_modulus': 7.5e6, #5e6, 7.5e6, 2e7
        'NU': 11.0,
        'n_control_points': 6,
        'alpha': 75.0,
        'max_rate_of_change_of_activation': np.inf,
        'target_v_max': 0.50,
        'p_static': 0.30,
        'boundary':(-0.35, 0.35, 0.90, 1.0, -0.35, 0.35),
        'final_time': 10.0,
        'success_threshold': 0.01,
        'w_dist': 2.0,
        'w_precision': 5.0,
        'w_progress': 1.0,
        'w_smoothness': 0.03,
        'w_on_target': 0.5,
        'sigma_mult': 1.5,
        'sigma_floor': 0.01,
    }

    # Output directory
    out_dir = Path("results_expert_E7.5e6_v28")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist env_params alongside the model for full reproducibility.
    sp = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
          for k, v in env_params.items()}
    sp['boundary'] = list(env_params['boundary'])
    with open(out_dir / "env_params.json", "w") as f:
        json.dump(sp, f, indent=2)

    print(f"\nYoung's modulus: {env_params['young_modulus']:.0e}")
    print(f"Mix: {env_params['p_static']:.0%} statico / "
          f"{1-env_params['p_static']:.0%} a {env_params['target_v_max']} m/s")
    print(f"sigma_near: {max(env_params['success_threshold'] * env_params['sigma_mult'], env_params['sigma_floor']):.4f} m")
    print(f"Total budget: {total_steps:,} steps (~{total_steps/1400/60:.0f} min)")

    # Vectorized env stack
    train_env = SubprocVecEnv(
        [make_env(i, seed, env_params) for i in range(n_envs)],
        start_method="spawn")
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(250_000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="sac_27",
        save_vecnormalize=True
    )

    logging_cb = LoggingCallback(
        out_dir=out_dir,
        eval_window=200,
        log_every_episodes=20,
        log_every_steps=10_000
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        target_entropy=-6.0,
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300])),
        verbose=1,
        tensorboard_log=str(out_dir / "tensorboard"),
        seed=seed
    )

    print(f"Starting SAC v27 for {total_steps:,} steps\n")

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, logging_cb],
        progress_bar=True
    )
    elapsed = time.time() - t0

    # Final save
    model.save(str(out_dir / "final_model"))
    train_env.save(str(out_dir / "vec_normalize_final.pkl"))
    train_env.close()

    print(f"\n{'='*70}")
    print(f" Training completed in {elapsed/60:.1f} min")
    print(f" Models saved in: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
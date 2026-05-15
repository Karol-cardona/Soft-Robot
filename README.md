# Adaptive Robot Control to Body Variations

Soft-robot reinforcement learning thesis project: train a single
control policy that adapts to runtime changes in body stiffness
(Young's modulus) without retraining.

> **University of Pisa** — Thesis project by **Karol**  
> **Tutors:** Enrico Donato, Francesca Sparnacci

---

## Motivation

Soft and compliant robots exhibit changes in dynamic properties
(stiffness, inertia, damping) due to environmental interactions,
wear, or temperature shifts. A controller tuned for one set of
properties degrades when those properties drift, and retraining on
the new body is rarely practical at deployment time.

This project trains a single neural-network controller that takes
the body's stiffness as an extra input and produces the appropriate
control action online, eliminating the need for re-training when
the body changes within the trained range.

The base task follows Case 1 of Naughton et al. (2021): a slender
Cosserat rod fixed at its base, actuated by spline-parameterized
internal torques, must track a randomly moving target in 3-D
space. We extend that setup with stiffness variation along
Young's modulus E in the range `[5e6, 2e7] Pa`.

## Methodology

The pipeline distills four expert SAC controllers (each trained at
a specific E) into one E-conditioned student via supervised
imitation, then refines the student through targeted fine-tuning
stages that address the two main failure modes of plain
Behavioral Cloning:

```
   4 expert SAC controllers           training_student.py +
   trained at E in {5e6, 7.5e6,        training_student_noise.py
   1e7, 2e7} Pa                     ┌──────────────┬──────────────┐
   (training.py)                    │              │              │
        │                           ▼              ▼              │
        │   generate_dataset    Student v3    Student v4_noise    │
        ▼                       (plain BC)    (E_norm noise       │
   distillation_dataset.npz         │          augmentation)      │
   ~2M (state, action,              │              │              │
   logit) pairs                     └──┐        ┌──┘              │
                                       ▼        ▼                 │
                                    dagger_training.ipynb         │
                                       │                          │
                                       ├── Stage 1: Standard      │
                                       │   DAgger (3 rounds)      │
                                       │   on 4 trained E values  │
                                       │                          │
                                       └── Stage 2: Targeted      │
                                           DAgger (2 rounds)      │
                                           includes 6.5e6, 9e6,   │
                                           1.5e7 (interpolated)   │
                                           labeled by nearest     │
                                           expert                 │
                                              │                   │
                                              ▼                   │
                                       Final policy:              │
                                       student_targeted_r1.pth ◀──┘
                                              │
                                              ▼
                                       validate_student.ipynb
                                       adaptation_test.py
```

### Failure modes addressed at each stage

| Stage                  | Failure mode                              | Fix                                                       |
|------------------------|-------------------------------------------|-----------------------------------------------------------|
| v3 — Behavioral Cloning | None (baseline)                          | -                                                         |
| v4_noise               | Step-function behavior across E (poor    | Gaussian noise (sigma=0.05) on E_norm during fine-tuning  |
|                        | interpolation to unseen stiffness)        |                                                           |
| Standard DAgger        | Distribution shift: student visits states| On-policy collection labeled by the matched expert        |
|                        | the experts never see                     |                                                           |
| Targeted DAgger        | Residual gap on interpolated E values     | Roll out AT interpolated stiffness; label with nearest    |
|                        | (6.5e6, 9e6, 1.5e7) not seen at training  | trained expert                                            |

The full sequence v3 → v4_noise → Standard DAgger → Targeted DAgger
moves performance @1cm by roughly +20 to +60 percentage points
depending on the stiffness regime, with the largest gains on the
interpolated points (which start near 14% under plain BC and end
above 70%).

## Final results

Selected model: **Targeted DAgger, Round 1**  
(`results_dagger_targeted/round_1/student_targeted_r1.pth`)

Round 2 was evaluated but over-specialized on the interpolation
regimes, regressing slightly on the trained experts. Round 1 was
selected for the final pipeline.

### Discrete E validation (pure dynamic, p_static=0.0)

| E (Pa)   | Type         | v3 BC | v4_noise | Std DAgger | **Targeted R1** | Δ vs v3   |
|----------|--------------|------:|---------:|-----------:|----------------:|----------:|
| 5e6      | trained      | 41%   | 41%      | 62%        | **61%**         | +20 pp    |
| 6.5e6    | interpolated | 14%   | 28%      | 52%        | **75%**         | +61 pp    |
| 7.5e6    | trained      | 35%   | 35%      | 59%        | **69%**         | +34 pp    |
| 9e6      | interpolated | 45%   | 50%      | 76%        | **77%**         | +32 pp    |
| 1e7      | trained      | 57%   | 58%      | 79%        | **82%**         | +25 pp    |
| 1.5e7    | interpolated | 32%   | 59%      | 87%        | **87%**         | +55 pp    |
| 2e7      | trained      | 72%   | 74%      | 91%        | **92%**         | +20 pp    |

Values are success rates @1cm (fraction of steps with tip-to-target
distance below 1 cm), averaged across 50 episodes per (E, model).
The 2e7 expert is matched by the final student (92% vs expert's
~93%), and the largest interpolation gains are on 6.5e6 (+61 pp)
and 1.5e7 (+55 pp), which are exactly the regimes plain BC could
not handle.

### Static-target validation (p_static=1.0)

The hardest regime, where the agent must hold the rod tip
perfectly still on a randomly placed target:

| Condition          | Targeted R1 success @1cm |
|--------------------|-------------------------:|
| 5e6  (soft)        | 10.6%                    |
| 6.5e6 (interp)     | 19.9%                    |
| 7.5e6              | 20.8%                    |
| 9e6  (interp)      | 18.1%                    |
| 1e7                | 23.5%                    |
| 1.5e7 (interp)     | 19.9%                    |
| 2e7  (rigid)       | 17.2%                    |

Comparison with v3 BC on rigid static: v3 → 0.6%, Targeted R1 →
17.2% (~30x improvement). Static is intrinsically harder than
dynamic at the same stiffness because there is no anticipatory
signal to exploit — pure hold-in-place behavior is required.

### Adaptation results

Tests the controller's response when its *belief* about stiffness
changes mid-episode while the physical rod stays at E=1e7. See
`adaptation_test.py` for the methodological caveat (PyElastica
cannot change E at runtime without rebuilding the rod, so we vary
the conditioning input rather than the physical stiffness).

| Scenario                       | Median recovery time | Pre/peak/post error (cm) | Degradation factor |
|--------------------------------|---------------------:|-------------------------:|-------------------:|
| Step **soft → stiff** (E_norm 0→1)  | 2 steps               | 0.79 / 1.11 / 0.79     | 1.39x              |
| Step **stiff → soft** (E_norm 1→0)  | 0 steps               | 0.98 / 0.79 / 0.79     | 1.22x              |
| Gradual drift (E_norm 0→1)          | -                     | first/mid/last thirds: 69% / 84% / 60% @1cm | -      |

The agent recovers immediately under stiff→soft transitions
(post-error actually IMPROVES over pre-error), and within 2 steps
under soft→stiff transitions. The gradual drift scenario shows
peak performance in the middle of the trajectory (E_norm ~ 0.5,
the most trained value), with mild degradation at the soft and
stiff ends.

## Repository layout

```
.
├── README.md                              ← this file
│
├── rod_tracking_env.py                    Cosserat rod environment (Gymnasium-compatible),
│                                          wraps PyElastica. Defines obs space (44 dims),
│                                          action space (12 dims), reward, dynamic/static
│                                          target sampling.
│
├── training.py                            Single-expert SAC training. Run four times
│                                          (one per Young's modulus value) with
│                                          CHANGE PER EXPERT markers documented in the
│                                          file header.
│
├── sanity_check_env.ipynb                 Standalone env tests (reset, step, action
│                                          bounds, reward components). Run after any
│                                          modification of rod_tracking_env.py.
│
├── validation_rod_tracking_all_experts.ipynb
│                                          Validate each trained expert under 3
│                                          evaluation conditions (matched / pure_dyn /
│                                          pure_static) and a spatial diagnostic.
│
├── generate_dataset.ipynb                 Roll out each expert and assemble the
│                                          distillation dataset (~2M (obs+E_norm,
│                                          action, logit) triples) for downstream BC.
│
├── training_student.py                    Student v3: plain Behavioral Cloning. MLP
│                                          [400, 300, 256] conditioned on E_norm,
│                                          outputs pre-tanh logits.
│
├── training_student_noise.py              v4_noise: fine-tunes v3 with Gaussian noise
│                                          on E_norm to enable smooth interpolation
│                                          between trained stiffness values.
│
├── dagger_training.ipynb                  Both DAgger stages: standard (3 rounds, 4
│                                          experts) and Targeted (2 rounds, includes
│                                          interpolations labeled by nearest expert).
│
├── validate_student.ipynb                 Full student validation: 3 conditions × 7
│                                          stiffness values + continuous generalization
│                                          test (100 random E values in log-space).
│
├── adaptation_test.py                     Step (soft→stiff, stiff→soft) and gradual
│                                          drift scenarios. Measures recovery time,
│                                          peak error, degradation factor.
│
├── classical_pd_controller.py             Bonus baseline: gain-scheduled adaptive PD
│                                          controller as a drop-in StudentPolicy
│                                          replacement.
│
├── tune_pd.py                             3-stage grid search for PD gains.
│
├── distillation_data/                     Generated by generate_dataset.ipynb
│   └── distillation_dataset.npz
│
├── results_expert_E5e6_v28/                ┐
├── results_expert_E7.5e6_v27/              │ Per-expert training output
├── results_expert_E1e7_v27/                │ (best_model.zip, vecnorm_best.pkl,
├── results_expert_E2e7_v27/                ┘  tensorboard logs)
│
├── results_student_v3/                     Student BC baseline
├── results_student_v4_noise/               Student after E_norm noise fine-tuning
├── results_dagger_fast/                    Standard DAgger output
└── results_dagger_targeted/                Targeted DAgger output
    ├── round_1/                            ← FINAL SELECTED MODEL
    │   ├── student_targeted_r1.pth
    │   └── validation/
    └── round_2/                            evaluated but not selected
```

## Setup

```bash
# Python 3.10 or later
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies

| Package            | Purpose                              |
|--------------------|--------------------------------------|
| PyElastica         | Cosserat-rod simulator               |
| Stable Baselines 3 | SAC expert training                  |
| PyTorch (+ CUDA)   | Student network training             |
| Gymnasium          | Env interface (used by RodTrackingEnv) |
| NumPy, Pandas      | Data handling                        |
| Matplotlib         | Plots                                |
| tqdm               | Progress bars                        |

A GPU is recommended for the student training (DAgger rounds run
in ~10 minutes per round on an RTX-3070-class GPU; multi-hour on
CPU). The Cosserat simulation itself is CPU-bound and parallelized
across 8 SubprocVecEnv workers during data collection.

## Reproducing the pipeline

Each stage can be run independently if the previous artifacts
exist on disk.

### 1. Expert training (4 runs)

Open `training.py`. The header lists CHANGE PER EXPERT markers
(`young_modulus` value and `out_dir`). Set them to the four
training values one at a time:

```bash
# E = 5e6, output -> results_expert_E5e6/
python training.py

# Edit training.py: young_modulus = 7.5e6, out_dir = "results_expert_E7.5e6"
python training.py

# ...repeat for 1e7 and 2e7
```

Each expert takes roughly 8-12 hours on a modern GPU to reach
convergence (target ~10M timesteps).

### 2. Sanity-check the env

After any modification to `rod_tracking_env.py`:

```bash
jupyter notebook sanity_check_env.ipynb
```

Runs through the env tests in order. All should pass before
proceeding.

### 3. Validate experts

```bash
jupyter notebook validation_rod_tracking_all_experts.ipynb
```

Produces per-expert results and cross-expert plots under
`Evaluation_v2/`. Useful to confirm that each expert reached a
reasonable success rate before distillation.

### 4. Generate the distillation dataset

```bash
jupyter notebook generate_dataset.ipynb
```

Rolls out each expert (150 dynamic + 50 static episodes), filters
on `on_goal_fraction`, saves the combined dataset to
`distillation_data/distillation_dataset.npz` (~2M samples).

### 5. Student v3 — Behavioral Cloning

```bash
python training_student.py
```

Output: `results_student_v3/student_policy.pth`.

### 6. Student v4_noise — Fine-tune with E_norm noise

```bash
python training_student_noise.py
```

Output: `results_student_v4_noise/student_policy.pth`.

### 7. DAgger pipeline (both stages)

```bash
jupyter notebook dagger_training.ipynb
```

Run the cells in order:
- Stage 1 (Standard DAgger, 3 rounds) outputs to `results_dagger_fast/`.
- Stage 2 (Targeted DAgger, 2 rounds) outputs to `results_dagger_targeted/`.

Total runtime ~1.5-2 hours on GPU.

### 8. Final validation

```bash
jupyter notebook validate_student.ipynb
```

By default validates the Targeted DAgger Round 1 student. To
validate a different checkpoint, uncomment the appropriate
`STUDENT_PATH` / `OUTPUT_DIR` block at the top of the notebook.

### 9. Adaptation test

```bash
python adaptation_test.py
```

Runs the three adaptation scenarios (step soft→stiff, step
stiff→soft, gradual drift) on the final student and writes results
under `results_dagger_targeted/round_1/validation/`.

## Bonus: classical baseline

`classical_pd_controller.py` implements a gain-scheduled adaptive
PD controller as a drop-in replacement for the neural-network
student. Both fixed-gain and adaptive variants are provided:

```
K_P(E) = K_P_base * (1 + alpha_P * E_norm)
K_D(E) = K_D_base * (1 + alpha_D * E_norm)
```

To validate the PD controller, duplicate `validate_student.ipynb`
and replace the `load_student` call with `load_pd_controller`. The
PD module is a `nn.Module` subclass with the same interface as
`StudentPolicy`, so the rest of the validation flow needs no
changes.

`tune_pd.py` performs a 3-stage grid search over (a) gain signs,
(b) base gains, (c) adaptive slopes.

This provides a classical adaptive-control baseline against which
the learned student can be directly compared. Following the
methodology of Åström & Wittenmark (Adaptive Control, 1995), this
is gain-scheduled adaptive control — a step less general than
full Model Reference Adaptive Control (MRAC), which would require
a reference model and online plant identification (impractical for
a Cosserat rod PDE).

## References

The methods build on the following key works:

1. **Naughton, Sun, Tekinalp, Parthasarathy, Chowdhary, Gazzola.**
   *Elastica: A Compliant Mechanics Environment for Soft Robotic
   Control.* IEEE RAL, 2021.  
   Establishes the Cosserat-rod environment used here. Our Case 1
   (3-D tracking of a moving target) follows their reward and
   action-space design closely.

2. **Haarnoja, Zhou, Abbeel, Levine.** *Soft Actor-Critic: Off-Policy
   Maximum Entropy Deep RL with a Stochastic Actor.* ICML, 2018.  
   SAC algorithm used for expert training.

3. **Ross, Gordon, Bagnell.** *A Reduction of Imitation Learning and
   Structured Prediction to No-Regret Online Learning.* AISTATS, 2011.  
   The DAgger algorithm.

4. **Yu, Tan, Liu, Turk.** *Preparing for the Unknown: Learning a
   Universal Policy with Online System Identification.* RSS, 2017.  
   Inspiration for conditioning a single policy on physical
   parameters (UPN).

5. **Kadokawa, Hamaya, et al.** *Cyclic Policy Distillation.*  2023.  
   Inspiration for the distillation framework with conditioning.

6. **Haldar et al.** *PolyTask.* 2023.  
   Related work on multi-task distillation through DAgger.

7. **Åström, Wittenmark.** *Adaptive Control* (2nd ed.). Addison-Wesley,
   1995.  
   Reference for the classical gain-scheduled PD baseline.

## Notes for future work

- **Real physical stiffness changes**: the current adaptation test
  varies the student's belief about E while the rod stays
  physically fixed. PyElastica does not support runtime
  modification of E without rebuilding the rod, which resets
  simulation state. Testing genuine physical stiffness changes
  would require either a custom rod-rebuilding scheme that
  preserves state, or migration to a different simulator.

- **Beyond stiffness**: the same conditioning approach extends to
  other physical parameters (density, damping coefficient,
  geometry). Each adds one dimension to the conditioning input
  but the pipeline is otherwise unchanged.

- **MRAC baseline**: a full Model Reference Adaptive Control
  baseline (with online plant identification) was out of scope
  for the Cosserat PDE plant. Future work could derive a
  reduced-order linear model of the rod tip dynamics and compare
  the learned policy against a proper MRAC scheme.

---

*Generated as part of the cleanup and documentation pass on the
project repository (May 2026).*

"""
POV-Ray Rendering Pipeline — Thesis Videos
==========================================

Generates all videos required for the thesis presentation:

  DYNAMIC (one per expert, moving target, E_norm = E_phys):
    - dynamic_E5e6.mp4
    - dynamic_E7p5e6.mp4
    - dynamic_E1e7.mp4
    - dynamic_E2e7.mp4

  STATIC (one per expert, stationary target):
    - static_E5e6.mp4
    - static_E7p5e6.mp4
    - static_E1e7.mp4
    - static_E2e7.mp4

  ADAPTATION (E changes during the episode via hot-swap; the rod's
              physical Young's modulus is updated whenever E_norm
              changes, keeping the student's belief and the physics
              synchronized):
    - drift_soft_to_stiff.mp4   (E: 5e6 → 2e7 at midpoint)
    - drift_stiff_to_soft.mp4   (E: 2e7 → 5e6 at midpoint)
    - drift_gradual.mp4         (E: 5e6 → 2e7 continuous linear ramp)

TEXT OVERLAY:
  - E_phys (Pa) shown instead of E_norm
  - Err (cm) with color coding: green<1cm, yellow<2cm, red>=2cm
  - Speed (m/s)
  - Status: ON TARGET / CLOSE / TRACKING
  - For drift scenarios: blinking "STEP CHANGE!" alert at the swap point

REQUIREMENTS:
  - POV-Ray installed
  - ffmpeg installed (with libass for subtitle rendering)
"""

import os
import argparse
import subprocess
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from rod_tracking_env import RodTrackingEnv
from training_student import StudentPolicy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ======================================================================
# CONFIGURATION
# ======================================================================

POVRAY_EXE = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine64.exe"
FFMPEG_EXE = "ffmpeg"

STUDENT_PATH = "results_dagger_targeted/round_1/student_targeted_r1.pth"
OUTPUT_BASE = Path("results_dagger_targeted/round_1/videos_final")

# Young's modulus range (must match the training/distillation convention)
E_MIN, E_MAX = 5e6, 2e7

# Expert stiffness values (one video per expert for dynamic/static)
E_VALUES = {
    "E5e6":   5e6,
    "E7p5e6": 7.5e6,
    "E1e7":   1e7,
    "E2e7":   2e7,
}

ENV_PARAMS_BASE = {
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

# Rendering settings
WIDTH = 1280
HEIGHT = 960
FRAME_STRIDE = 24        # Render every Nth simulation step (controls video length)
FPS = 30
PAPER_SCALE = 3.0        # Scale factor matching the Naughton et al. paper figures


# ======================================================================
# UTILITIES
# ======================================================================

def normalize_E(E):
    """Map physical Young's modulus to normalized [0, 1] (log-scale)."""
    return (np.log10(E) - np.log10(E_MIN)) / (np.log10(E_MAX) - np.log10(E_MIN))


def denormalize_E(E_norm):
    """Inverse of normalize_E: maps E_norm in [0, 1] back to physical E (Pa)."""
    log_min = np.log10(E_MIN)
    log_max = np.log10(E_MAX)
    return 10 ** (log_min + E_norm * (log_max - log_min))


def load_student(path, device):
    """Load the trained student policy from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StudentPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def student_predict(model, obs, E_norm, device):
    """Run the student forward with `obs` augmented by E_norm."""
    obs_aug = np.concatenate([obs, [E_norm]]).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
        action = torch.tanh(model(x)).squeeze(0).cpu().numpy()
    return action


# ======================================================================
# SIMULATION
# ======================================================================

def run_episode_collect(model, device, E_initial, E_norm_schedule, p_static, seed):
    """
    Run a single episode and collect trajectory data for rendering.

    The physical Young's modulus is kept synchronized with the student's
    E_norm input: whenever the schedule produces a meaningfully different
    E_norm, the rod's bend/shear matrices are hot-swapped to match.
    This guarantees that the belief used by the student and the actual
    physics of the rod stay coherent throughout the episode.

    Parameters
    ----------
    model : StudentPolicy
        Trained student.
    device : torch.device
    E_initial : float
        Initial physical Young's modulus (used to instantiate the rod).
    E_norm_schedule : callable
        E_norm_schedule(step) -> E_norm in [0, 1]. Defines how the
        stiffness should evolve over the episode.
    p_static : float
        Probability that the target is static for this episode (0=moving, 1=static).
    seed : int
        RNG seed for the environment.

    Returns
    -------
    traj : dict
        Per-step arrays: rod_positions, rod_radii, target_positions,
        errors, tip_speeds, E_norms.
    """
    env_params = {**ENV_PARAMS_BASE,
                  'young_modulus': E_initial,
                  'p_static': p_static}
    env = RodTrackingEnv(**env_params)
    obs, _ = env.reset(seed=seed)

    traj = {
        'rod_positions': [],
        'rod_radii': [],
        'target_positions': [],
        'errors': [],
        'tip_speeds': [],
        'E_norms': [],
    }

    step = 0
    done = False
    prev_tip = None
    last_E_phys = E_initial
    dt_env = ENV_PARAMS_BASE['sim_dt'] * ENV_PARAMS_BASE['num_steps_per_update']

    while not done:
        E_norm_now = E_norm_schedule(step)

        # Hot-swap the physical Young's modulus whenever E_norm changes
        # meaningfully (>0.1% relative change). For static/dynamic scenarios
        # the schedule is constant and this branch never fires.
        E_target = denormalize_E(E_norm_now)
        if abs(E_target - last_E_phys) / last_E_phys > 1e-3:
            env.change_young_modulus(E_target)
            last_E_phys = E_target

        action = student_predict(model, obs, E_norm_now, device)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        rod = env.shearable_rod
        current_tip = rod.position_collection[:, -1].copy()

        # Finite-difference tip speed (skip first step where we have no prev_tip)
        if prev_tip is not None:
            tip_speed = np.linalg.norm(current_tip - prev_tip) / dt_env
        else:
            tip_speed = 0.0
        prev_tip = current_tip

        traj['rod_positions'].append(rod.position_collection.copy())
        traj['rod_radii'].append(rod.radius.copy())
        traj['target_positions'].append(info['target_pos'].copy())
        traj['errors'].append(info['error'])
        traj['tip_speeds'].append(tip_speed)
        traj['E_norms'].append(E_norm_now)
        step += 1

    env.close()
    for k in ['rod_positions', 'rod_radii', 'target_positions', 'errors',
              'tip_speeds', 'E_norms']:
        traj[k] = np.array(traj[k])
    return traj


# ======================================================================
# POV-RAY SCENE GENERATION (paper-exact visual style)
# ======================================================================

def pov_camera_and_lights():
    """Camera, lights and background — matched to the Naughton et al. paper style."""
    return """
#version 3.7;
global_settings { assumed_gamma 1.0 }
#default { finish { ambient 0.1 diffuse 0.9 } }

#include "colors.inc"
#include "textures.inc"
#include "metals.inc"

// Camera positioned slightly closer than the original paper for clearer detail
#declare Camera_Position = <30.00, 9.00, -38.00>;
#declare Camera_Look_At  = <-2.00, 5.50, 8.00>;
#declare Camera_Angle    = 32;

camera {
    location Camera_Position
    right    x*image_width/image_height
    angle    Camera_Angle
    look_at  Camera_Look_At
}

light_source { <1500, 2500, -1000> color White }
light_source { Camera_Position color rgb <0.9, 0.9, 1> * 0.1 }

background { color White }

plane { <0, 1.5, 0>, -0.1
    texture { pigment { color White*1.1 } }
}
"""


def pov_rod_paper(positions, radii):
    """Build a sphere_sweep representing the rod in POV-Ray."""
    n_elems = positions.shape[1]
    lines = [f"sphere_sweep\n{{ linear_spline {n_elems}"]
    for i in range(n_elems):
        x, y, z = positions[:, i]
        r = radii[i] if i < len(radii) else radii[-1]
        # World coords → POV coords (axis remapping + scaling)
        sx = PAPER_SCALE * z
        sy = PAPER_SCALE * x
        sz = PAPER_SCALE * y
        sr = PAPER_SCALE * r
        lines.append(f",\n<{sx:.4f}, {sy:.4f}, {sz:.4f}>, {sr:.4f}")
    lines.append("\ntexture {")
    lines.append("    pigment { color rgb <0.45, 0.39, 1> transmit 0.1 }")
    lines.append("    finish { phong 1 }")
    lines.append("}")
    lines.append("scale <4, 4, 4> rotate <0, 90, 90> translate <2, 0, 4>")
    lines.append("}")
    return "\n".join(lines)


def pov_base_paper(radii):
    """Build the gray sphere representing the fixed base of the rod."""
    r_rod = radii[0] if len(radii) > 0 else 0.05
    base_r = 1.5 * PAPER_SCALE * r_rod
    return f"""
sphere {{
    <0, 0, 0>, {base_r:.4f}
    texture {{
        pigment {{ color rgb <0.75, 0.75, 0.75> transmit 0.1 }}
        finish {{ phong 1 }}
    }}
    scale <4, 4, 4> rotate <0, 90, 90> translate <2, 0, 4>
}}
"""


def pov_target_paper(position, rod_radii):
    """Build the orange sphere representing the moving target."""
    x, y, z = position
    rod_base_r = rod_radii[0] if len(rod_radii) > 0 else 0.05
    target_r = PAPER_SCALE * rod_base_r * 1.5

    sx = PAPER_SCALE * z
    sy = PAPER_SCALE * x
    sz = PAPER_SCALE * y

    return f"""
sphere {{
    <{sx:.4f}, {sy:.4f}, {sz:.4f}>, {target_r:.4f}
    texture {{
        pigment {{ color rgb <1, 0.5, 0.4> transmit 0.1 }}
        finish {{ phong 1 }}
    }}
    scale <4, 4, 4> rotate <0, 90, 90> translate <2, 0, 4>
}}
"""


def generate_pov_file(frame_path, rod_pos, rod_radii, target_pos):
    """Assemble a complete .pov scene file for one frame."""
    scene = (
            pov_camera_and_lights()
            + pov_base_paper(rod_radii)
            + pov_rod_paper(rod_pos, rod_radii)
            + pov_target_paper(target_pos, rod_radii)
    )
    with open(frame_path, 'w') as f:
        f.write(scene)


# ======================================================================
# RENDERING + TEXT OVERLAY
# ======================================================================

def render_frame(pov_path, png_path):
    """Render a single .pov file to PNG via POV-Ray."""
    pov_abs = str(Path(pov_path).resolve())
    png_abs = str(Path(png_path).resolve())
    cmd = [
        POVRAY_EXE,
        f"+I{pov_abs}",
        f"+O{png_abs}",
        f"+W{WIDTH}",
        f"+H{HEIGHT}",
        "+A0.3",        # Anti-aliasing threshold
        "+AM2",         # Anti-aliasing method
        "+Q11",         # Quality level
        "+FN",          # PNG output
        "/EXIT",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Fallback: re-force linear_spline (defensive, scene already uses it)
        with open(pov_abs, 'r') as f:
            content = f.read()
        content = content.replace("b_spline", "linear_spline")
        with open(pov_abs, 'w') as f:
            f.write(content)
        subprocess.run(cmd, check=True, capture_output=True)


def fmt_time(t):
    """Format seconds as h:mm:ss.cc for ASS subtitle timestamps."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_ass_subtitles(ass_file, traj, frame_indices, scenario_name,
                           change_step=None):
    """
    Generate an ASS subtitle file with per-frame overlay text:
      Top:    E_phys, error (color-coded), tip speed
      Bottom: status (ON TARGET / CLOSE / TRACKING)
      Alert:  blinking "STEP CHANGE!" near the swap point (drift only)
    """
    header = """[Script Info]
Title: Rod Tracking Overlay
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 960
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Top,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1,8,20,20,25,1
Style: Bottom,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1,2,20,20,25,1
Style: Alert,Arial,40,&H000000FF,&H000000FF,&H00FFFFFF,&H80000000,1,0,0,0,100,100,0,0,1,3,2,5,20,20,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # ASS subtitle color codes (BGR format, &H prefix)
    COLOR_WHITE  = "{\\c&HFFFFFF&}"
    COLOR_GREEN  = "{\\c&H00FF00&}"
    COLOR_YELLOW = "{\\c&H00FFFF&}"
    COLOR_RED    = "{\\c&H0000FF&}"
    ALERT_FX     = "{\\c&H0000FF&\\fad(100,100)}"  # Red alert with fade in/out

    frame_duration = 1.0 / FPS
    events = []

    # Map change_step (simulation step) to frame index in the rendered video
    change_frame = None
    if change_step is not None:
        for i, step in enumerate(frame_indices):
            if step >= change_step:
                change_frame = i
                break

    for i, step in enumerate(frame_indices):
        t_start = i * frame_duration
        t_end = (i + 1) * frame_duration

        err_cm = traj['errors'][step] * 100
        speed = traj['tip_speeds'][step]
        E_norm = traj['E_norms'][step]
        E_phys = denormalize_E(E_norm)

        # Color-code the error display by precision level
        if err_cm < 1.0:
            err_color = COLOR_GREEN
            status = "ON TARGET"
            status_color = COLOR_GREEN
        elif err_cm < 2.0:
            err_color = COLOR_YELLOW
            status = "CLOSE"
            status_color = COLOR_YELLOW
        else:
            err_color = COLOR_RED
            status = "TRACKING"
            status_color = COLOR_RED

        # Top line: main parameters (E, error, tip speed)
        text_top = (
                COLOR_WHITE
                + f"E = {E_phys:.2e} Pa  |  "
                + f"Err: {err_color}{err_cm:.1f} cm{COLOR_WHITE}  |  "
                + f"Speed: {speed:.2f} m/s"
        )

        # Bottom line: status label
        text_bottom = f"{status_color}{status}"

        events.append(
            f"Dialogue: 0,{fmt_time(t_start)},{fmt_time(t_end)},Top,,0,0,0,,{text_top}"
        )
        events.append(
            f"Dialogue: 0,{fmt_time(t_start)},{fmt_time(t_end)},Bottom,,0,0,0,,{text_bottom}"
        )

        # Blinking alert around the change point (step scenarios only)
        if change_frame is not None and abs(i - change_frame) < int(FPS * 0.75):
            alert_text = ALERT_FX + ">>> STEP CHANGE! <<<"
            events.append(
                f"Dialogue: 1,{fmt_time(t_start)},{fmt_time(t_end)},Alert,,0,0,0,,{alert_text}"
            )

    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))


def add_text_overlay(input_video, output_video, ass_file):
    """Burn the ASS subtitles into the video using ffmpeg+libass."""
    # ffmpeg requires the path escaped for the ass filter
    ass_path_ffmpeg = str(ass_file).replace("\\", "/").replace(":", r"\:")
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", str(input_video),
        "-vf", f"ass='{ass_path_ffmpeg}'",
        "-c:v", "libx264",
        "-crf", "18",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def render_trajectory(traj, scenario_name, out_dir, change_step=None):
    """
    End-to-end rendering pipeline for one trajectory:
      1. Generate .pov scene files (one per frame)
      2. Render each .pov to PNG via POV-Ray
      3. Compose PNGs into a raw .mp4 with ffmpeg
      4. Burn the text overlay (ASS subtitles) into the final video
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    pov_dir = out_dir / "pov_files"
    pov_dir.mkdir(exist_ok=True)

    n_steps = len(traj['rod_positions'])
    frame_indices = list(range(0, n_steps, FRAME_STRIDE))

    # Step 1: generate .pov scene files
    print(f"  Generating {len(frame_indices)} .pov files...")
    for i, step in enumerate(tqdm(frame_indices, desc="    POV")):
        pov_path = pov_dir / f"frame_{i:05d}.pov"
        generate_pov_file(
            pov_path,
            traj['rod_positions'][step],
            traj['rod_radii'][step],
            traj['target_positions'][step],
        )

    # Step 2: render PNGs via POV-Ray
    print("  Rendering with POV-Ray...")
    n_failed = 0
    for i in tqdm(range(len(frame_indices)), desc="    Render"):
        pov_path = pov_dir / f"frame_{i:05d}.pov"
        png_path = frames_dir / f"frame_{i:05d}.png"
        if png_path.exists():
            continue
        try:
            render_frame(pov_path, png_path)
        except subprocess.CalledProcessError:
            n_failed += 1
            continue
    if n_failed > 0:
        print(f"  {n_failed} frames failed (skipped)")

    # Step 3: compose raw video from PNG sequence
    video_raw = out_dir / f"{scenario_name}_raw.mp4"
    print("  Composing raw video...")
    cmd = [
        FFMPEG_EXE, "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(video_raw),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Step 4: subtitles + final overlay
    print("  Adding text overlay...")
    ass_file = out_dir / "overlay.ass"
    generate_ass_subtitles(ass_file, traj, frame_indices, scenario_name,
                           change_step=change_step)
    video_path = out_dir / f"{scenario_name}.mp4"
    add_text_overlay(video_raw, video_path, ass_file)

    # Clean up the intermediate raw file
    if video_raw.exists():
        video_raw.unlink()

    print(f"  Saved: {video_path}")


# ======================================================================
# SCENARIOS
# ======================================================================

def get_episode_length():
    """Number of RL steps per episode at the configured timings."""
    return int(ENV_PARAMS_BASE['final_time'] /
               (ENV_PARAMS_BASE['sim_dt'] * ENV_PARAMS_BASE['num_steps_per_update']))


def scenario_dynamic(model, device, E_physical, label):
    """One expert tracking a moving target at fixed E."""
    E_norm = normalize_E(E_physical)
    traj = run_episode_collect(
        model, device, E_physical,
        E_norm_schedule=lambda s: E_norm,
        p_static=0.0, seed=42,
    )
    return traj, f"dynamic_{label}", None


def scenario_static(model, device, E_physical, label):
    """One expert holding position on a stationary target at fixed E."""
    E_norm = normalize_E(E_physical)
    traj = run_episode_collect(
        model, device, E_physical,
        E_norm_schedule=lambda s: E_norm,
        p_static=1.0, seed=42,
    )
    return traj, f"static_{label}", None


def scenario_drift_soft_to_stiff(model, device):
    """Step change soft → stiff at the episode midpoint (physical hot-swap)."""
    n = get_episode_length()
    change_step = n // 2
    schedule = lambda s: 0.0 if s < change_step else 1.0
    E_initial = denormalize_E(schedule(0))   # = 5e6 (soft)

    traj = run_episode_collect(
        model, device, E_initial,
        E_norm_schedule=schedule, p_static=0.0, seed=42,
    )
    return traj, "drift_soft_to_stiff", change_step


def scenario_drift_stiff_to_soft(model, device):
    """Step change stiff → soft at the episode midpoint (physical hot-swap)."""
    n = get_episode_length()
    change_step = n // 2
    schedule = lambda s: 1.0 if s < change_step else 0.0
    E_initial = denormalize_E(schedule(0))   # = 2e7 (stiff)

    traj = run_episode_collect(
        model, device, E_initial,
        E_norm_schedule=schedule, p_static=0.0, seed=42,
    )
    return traj, "drift_stiff_to_soft", change_step


def scenario_drift_gradual(model, device):
    """Linear continuous drift from soft to stiff over the entire episode."""
    n = get_episode_length()
    schedule = lambda s: min(s / n, 1.0)
    E_initial = denormalize_E(schedule(0))   # = 5e6 (starts soft)

    traj = run_episode_collect(
        model, device, E_initial,
        E_norm_schedule=schedule, p_static=0.0, seed=42,
    )
    return traj, "drift_gradual", None


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["dynamic", "static", "drift", "all"],
                        default="all",
                        help="Which category of videos to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_student(STUDENT_PATH, device)
    print(f"Student loaded: {STUDENT_PATH}\n")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Build the list of scenarios to run based on --category
    all_scenarios = []

    if args.category in ("dynamic", "all"):
        for label, E in E_VALUES.items():
            all_scenarios.append(
                (f"DYNAMIC E={E:.1e}",
                 lambda m, d, e=E, l=label: scenario_dynamic(m, d, e, l))
            )

    if args.category in ("static", "all"):
        for label, E in E_VALUES.items():
            all_scenarios.append(
                (f"STATIC E={E:.1e}",
                 lambda m, d, e=E, l=label: scenario_static(m, d, e, l))
            )

    if args.category in ("drift", "all"):
        all_scenarios.extend([
            ("DRIFT soft → stiff", scenario_drift_soft_to_stiff),
            ("DRIFT stiff → soft", scenario_drift_stiff_to_soft),
            ("DRIFT gradual",      scenario_drift_gradual),
        ])

    print(f"Total videos to generate: {len(all_scenarios)}\n")

    for idx, (name, fn) in enumerate(all_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(all_scenarios)}]  {name}")
        print(f"{'='*60}")
        try:
            traj, label, change_step = fn(model, device)
            out_dir = OUTPUT_BASE / label
            render_trajectory(traj, label, out_dir, change_step=change_step)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll videos saved to: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
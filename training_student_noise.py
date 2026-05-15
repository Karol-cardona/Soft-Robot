"""
Student policy fine-tuning — noise augmentation on E_norm.

Builds on the v3 student (`training_student.py`) by fine-tuning with
Gaussian noise applied to the `E_norm` input dimension. The goal is
to convert a piecewise-discrete policy (4 training stiffness values)
into a smooth function of stiffness that can interpolate to values
unseen during training (e.g. 6.5e6, 9e6, 1.5e7).

Why this is needed
------------------
The base student is trained on a dataset where `E_norm` only ever
takes 4 distinct values:
    {0.000, 0.293, 0.500, 1.000}
corresponding to E in {5e6, 7.5e6, 1e7, 2e7}. Plain BC therefore
fits a step function on E_norm — perfect on the training values, but
arbitrary in between. In practice the v3 student reaches around 14%
@1cm at the interpolated stiffness 6.5e6 despite scoring 41% / 35% on
the adjacent training values.

Adding Gaussian noise to `E_norm` during fine-tuning (sigma=0.05,
~1/4 of the spacing between consecutive experts) forces the network
to produce similar outputs at neighboring E_norm values, smoothing
the conditioning function without erasing inter-expert differences.

Strategy
--------
- Start from the v3 checkpoint (already trained ~400 epochs).
- Fine-tune at a lower LR (1e-4 vs 3e-4) for 200 epochs max.
- Add Gaussian noise (sigma=0.05) to `E_norm` per-sample, clamped to
  [0, 1].
- Validate WITHOUT noise on E_norm — we want to measure the model on
  the exact 4 training values, not on the augmented distribution.

Note on val loss comparability
------------------------------
v4_noise's val loss is NOT directly comparable to v3's: v3 trains and
validates on the same (noise-free) distribution, while v4_noise
trains with E_norm noise but validates without it. The mild
distribution shift inflates v4_noise's val loss by a few percent,
which is the price paid for interpolation smoothness. Real
performance must be assessed downstream by running the student in
the env (see `validate_student.ipynb`).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from training_student import StudentPolicy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from training_student import (
    StudentPolicy,
    E_VALUES,
    E_PALETTE,
    extract_architecture,
)

# CONFIGURATION
BATCH_SIZE      = 512
EPOCHS          = 200
LEARNING_RATE   = 1e-4          # Fine-tuning: più basso del training iniziale
VAL_FRACTION    = 0.20
EARLY_STOP_PAT  = 25
OBS_NOISE_STD   = 0.002
E_NORM_NOISE_STD = 0.05         # NUOVO: noise Gaussiano su E_norm
GRAD_CLIP       = 1.0

DATASET_FILE    = "distillation_data/distillation_dataset.npz"
INITIAL_MODEL   = "results_student_v3/student_policy.pth"       # modello di partenza
OUTPUT_DIR      = "results_student_v4_noise"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/student_policy.pth"
PLOTS_DIR       = f"{OUTPUT_DIR}/plots"

E_MIN    = 5e6
E_MAX    = 2e7

# UTILITIES
def normalize_E(E: float) -> float:
    """Log-scale normalize E to [0, 1] (same definition as v3)."""
    return (np.log10(E) - np.log10(E_MIN)) / (np.log10(E_MAX) - np.log10(E_MIN))


def load_dataset(path: str):
    """Load (X, y_logit) tensors from the distillation .npz.

    Prefers `y_logit` (saved by generate_dataset v2+). Falls back to
    `atanh(y_action)` clamped to a safe range if only post-tanh
    actions are available.
    """
    data = np.load(path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    if 'y_logit' in data:
        y = torch.tensor(data['y_logit'], dtype=torch.float32)
        print("  Target: y_logit (pre-tanh)")
    else:
        y_action = torch.tensor(data['y'], dtype=torch.float32)
        y = torch.atanh(y_action.clamp(-0.9999, 0.9999)).clamp(-4.0, 4.0)
        print("  Target: atanh(y_action) fallback")
    print(f"Dataset: X={X.shape}, y={y.shape}")
    return X, y


def get_expert_mask(X: torch.Tensor, E: float) -> torch.Tensor:
    """Boolean mask selecting rows whose E_norm matches `E`."""
    E_norm_target = float(normalize_E(E))
    return (X[:, -1] - E_norm_target).abs() < 1e-4


def compute_per_expert_metrics(model, X_val, y_val_logit, device):
    """Per-expert val MSE in both logit and action spaces.

    Important: this evaluation uses CLEAN E_norm (no noise). We want
    to measure how well the fine-tuned model still performs at the
    exact 4 training stiffness values; smoothness on interpolations
    is assessed separately via the env validation notebook.
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for E in E_VALUES:
            mask = get_expert_mask(X_val, E)
            if mask.sum() == 0:
                continue
            Xm = X_val[mask].to(device)
            ym = y_val_logit[mask].to(device)
            logit_pred = model(Xm)
            logit_mse = nn.functional.mse_loss(logit_pred, ym).item()
            action_mse = nn.functional.mse_loss(
                torch.tanh(logit_pred), torch.tanh(ym)
            ).item()
            results[E] = {"logit": logit_mse, "action": action_mse}
    return results


def load_initial_model(path: str, obs_dim: int, action_dim: int, device):
    """Load v3 weights into a fresh StudentPolicy for fine-tuning.

    `weights_only=False` is set explicitly because we're loading a
    TRUSTED local checkpoint (saved by us); newer torch versions warn
    about this for security reasons but the warning isn't applicable
    here.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StudentPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded initial model: ep {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.5f}")
    return model


# TRAINING
def train_student_with_E_noise():
    """Fine-tune the v3 student with E_norm noise augmentation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    #  Dataset
    X_full, y_full = load_dataset(DATASET_FILE)
    obs_dim = X_full.shape[1] - 1
    action_dim = y_full.shape[1]

    # Train / val split (same seed as v3 -> consistent split)
    n_total = len(X_full)
    n_val = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val

    dataset = TensorDataset(X_full, y_full)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=True)

    X_val_full = X_full[val_ds.indices]
    y_val_full = y_full[val_ds.indices]

    print(f"\nTrain: {n_train:,} | Val: {n_val:,}")

    # Model / optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_initial_model(INITIAL_MODEL, obs_dim, action_dim, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = extract_architecture(model)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Architecture        : {arch}")
    print(f"E_norm noise        : σ={E_NORM_NOISE_STD}")
    print(f"Starting fine-tuning with E_norm noise augmentation...\n")

    train_losses = []
    val_losses = []
    per_expert_logit = {E: [] for E in E_VALUES}
    per_expert_action = {E: [] for E in E_VALUES}

    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_count = 0

    for epoch in range(EPOCHS):

        # Train
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Noise on the 44 obs columns (same as v3).
            obs_noise = torch.randn(batch_X.shape[0], obs_dim, device=device) * OBS_NOISE_STD
            noisy_obs = batch_X[:, :obs_dim] + obs_noise

            # Gaussian noise on the E_norm column
            E_norm_orig = batch_X[:, obs_dim:obs_dim+1]
            E_noise = torch.randn(batch_X.shape[0], 1, device=device) * E_NORM_NOISE_STD
            noisy_E = torch.clamp(E_norm_orig + E_noise, 0.0, 1.0)

            noisy_X = torch.cat([noisy_obs, noisy_E], dim=1)

            optimizer.zero_grad()
            pred = model(noisy_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        scheduler.step()

        # Validation (no noise — see compute_per_expert_metrics)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_X)
                val_loss += criterion(pred, batch_y).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # Per-expert metrics (always on clean E_norm).
        expert_metrics = compute_per_expert_metrics(
            model, X_val_full, y_val_full, device
        )
        for E in E_VALUES:
            m = expert_metrics.get(E, {"logit": float('nan'), "action": float('nan')})
            per_expert_logit[E].append(m["logit"])
            per_expert_action[E].append(m["action"])

        # Best model selection on clean val loss.
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            early_stop_count = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'target_space': 'logit',
                'E_norm_noise': E_NORM_NOISE_STD,
            }, MODEL_SAVE_PATH)
        else:
            early_stop_count += 1

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            exp_str = "  ".join(
                f"E{E:.0e}: act={expert_metrics.get(E,{}).get('action', float('nan')):.4f}"
                for E in E_VALUES
            )
            print(f"Ep {epoch+1:>3d}/{EPOCHS} | "
                  f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
                  f"LR: {current_lr:.2e} | {exp_str}")

        if early_stop_count >= EARLY_STOP_PAT:
            print(f"\nEarly stop ep {epoch+1} "
                  f"(best val={best_val_loss:.6f} @ ep {best_epoch})")
            break

    print(f"\nBest model: ep {best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"Saved to: {MODEL_SAVE_PATH}")

    # Summary
    summary = {
        "version": "v4_E_noise",
        "initial_model": INITIAL_MODEL,
        "E_norm_noise_std": E_NORM_NOISE_STD,
        "obs_noise_std": OBS_NOISE_STD,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "per_expert_final": {
            f"E_{E:.0e}": {
                "logit_mse": per_expert_logit[E][-1],
                "action_mse": per_expert_action[E][-1],
            } for E in E_VALUES
        },
    }
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUTPUT_DIR}/training_summary.json")

    _plot_losses(train_losses, val_losses,
                 per_expert_logit, per_expert_action, best_epoch)

# Plotting
def _plot_losses(train_losses, val_losses,
                 per_expert_logit, per_expert_action, best_epoch):
    """Three-panel summary: global loss + per-expert in both spaces."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: global loss
    ax = axes[0]
    ax.plot(epochs, train_losses, label='Train', color='steelblue')
    ax.plot(epochs, val_losses, label='Val', color='tomato')
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
               label=f'Best (ep {best_epoch})')
    ax.set_title(f"Fine-tuning with E_norm noise (σ={E_NORM_NOISE_STD})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (logit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel 2: per-expert val loss (logit space)
    ax = axes[1]
    for E in E_VALUES:
        ax.plot(epochs, per_expert_logit[E], label=f"E={E:.0e}", color=E_PALETTE[E])
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
    ax.set_title("Val Loss per esperto (logit)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (logit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel 3: per-expert val loss (action space)
    ax = axes[2]
    for E in E_VALUES:
        ax.plot(epochs, per_expert_action[E], label=f"E={E:.0e}", color=E_PALETTE[E])
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
    ax.set_title("Per-expert val loss (logit)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (action)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    path = f"{PLOTS_DIR}/distillation_loss_v4_noise.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    train_student_with_E_noise()
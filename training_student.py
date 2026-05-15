"""
Student policy training — policy distillation.

Distills four SAC experts trained at different Young's modulus values
(E in {5e6, 7.5e6, 1e7, 2e7} Pa) into a single MLP student conditioned
on the normalized stiffness `E_norm` (the 45th feature of the input).

Input  : 45-D vector (obs_44 + E_norm)
Output : 12-D logit vector (pre-tanh). At inference apply `tanh()`
         outside the model to obtain the action in [-1, 1].

Design notes
------------
1. Target = pre-tanh logits, NOT post-tanh actions.
   Roughly 11% of expert actions sit in tanh's saturation zone
   (|a| > 0.95) where gradient vanishes. Regressing on logits gives a
   dense, well-defined gradient everywhere. If the dataset .npz
   doesn't carry `y_logit`, we fall back to `atanh(y_action)` clamped
   to a safe range.

2. CosineAnnealingLR (not ReduceLROnPlateau).
   ReduceLROnPlateau interacts badly with SB3's Adam state when
   resuming from checkpoints, and tends to cut the LR too early
   when val loss has natural oscillations. Cosine decay is monotone
   and predictable.

3. Observation noise on obs only, NOT on E_norm.
   E_norm has 4 discrete training values (one per expert). Adding
   noise to it would teach the student to interpolate between
   stiffness values — that's intentionally left to a separate
   noise-fine-tuning step (training_student_noise.py).

4. Last-layer orthogonal init with gain=0.01.
   Keeps initial logits near 0, so initial actions are near 0
   (tanh(0)=0): the student starts neutral, not aggressively
   committed to a direction.

5. Per-expert weighted loss.
   `EXPERT_WEIGHTS` lets you up-weight underperforming experts. With
   all weights at 1.0 (current default), `weighted_mse` reduces to
   plain MSE — the machinery is in place but disabled.
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CONFIGURATION
BATCH_SIZE      = 512
EPOCHS          = 400
LEARNING_RATE   = 3e-4
VAL_FRACTION    = 0.20
EARLY_STOP_PAT  = 30
OBS_NOISE_STD   = 0.002         # Gaussian noise std on obs
GRAD_CLIP       = 1.0           # max gradient norm

# Per-expert sample weights. All 1.0 = uniform;
EXPERT_WEIGHTS  = {5e6: 1.0, 7.5e6: 1.0, 1e7: 1.0, 2e7: 1.0}

DATASET_FILE    = "distillation_data/distillation_dataset.npz"
OUTPUT_DIR      = "results_student_v3"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/student_policy.pth"
PLOTS_DIR       = f"{OUTPUT_DIR}/plots"

# E range — must match `generate_dataset.ipynb`.
E_MIN    = 5e6
E_MAX    = 2e7
E_VALUES = [5e6, 7.5e6, 1e7, 2e7]

E_PALETTE = {
    5e6:   "#e74c3c",   # red    — softest
    7.5e6: "#f39c12",   # orange
    1e7:   "#3498db",   # blue
    2e7:   "#27ae60",   # green  — stiffest
}

class StudentPolicy(nn.Module):
    """MLP conditioned on E_norm. Outputs PRE-tanh logits (not actions).

   Architecture: [hidden=400, 300, 256], LayerNorm + ReLU between
   layers, no activation on the output.

   At inference time the user is expected to apply `tanh()` to the
   output (or call `predict_action()`).
   """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        input_dim = obs_dim + 1   # obs_44 + E_norm

        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init. The OUTPUT layer uses gain=0.01 so that
        initial logits are near zero — i.e. tanh(logit) starts near 0
        and the student doesn't begin training with strongly committed
        actions.
        """
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                is_last = (i == len(self.net) - 1)
                gain = 0.01 if is_last else nn.init.calculate_gain('relu')
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return PRE-tanh logits. Apply tanh externally for inference."""
        return self.net(x)

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference helper: returns post-tanh actions."""
        return torch.tanh(self.net(x))

def extract_architecture(model: nn.Module) -> list:
    """Inspect a `StudentPolicy` and return its hidden layer sizes.

    Avoids hardcoding the architecture in the summary JSON — any
    future tweak to the model is automatically reflected.
    """
    sizes = []
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    # All but the last linear's `out_features` are hidden sizes.
    for layer in linear_layers[:-1]:
        sizes.append(layer.out_features)
    return sizes


def normalize_E(E: float) -> float:
    """Log-scale normalize a Young's modulus value to [0, 1]."""
    log_E   = np.log10(E)
    log_min = np.log10(E_MIN)
    log_max = np.log10(E_MAX)
    return (log_E - log_min) / (log_max - log_min)


def load_dataset(path: str):
    """Load the distillation dataset.

    Strategy for the regression target (in preference order):
      1. `y_logit` from the .npz   — preferred, saved by `generate_dataset` v2+
      2. `atanh(y_action)`         — fallback for legacy .npz files
    """
    data = np.load(path)
    X = torch.tensor(data['X'], dtype=torch.float32)

    if 'y_logit' in data:
        y = torch.tensor(data['y_logit'], dtype=torch.float32)
        print("  Target: y_logit (pre-tanh, dal .npz)")
    else:
        # Legacy fallback: convert post-tanh actions back to logits.
        # Clamp tightly before atanh to avoid infinities at +/-1.
        y_action = torch.tensor(data['y'], dtype=torch.float32)
        y_action_c = y_action.clamp(-0.9999, 0.9999)
        y = torch.atanh(y_action_c)
        # Cap the resulting logit magnitude; tanh(4) ~= 0.9993 so the
        # action is essentially preserved.
        y = y.clamp(-4.0, 4.0)
        print(f"  Logit stats: mean={y.mean():.3f}, std={y.std():.3f}, "
              f"max_abs={y.abs().max():.3f}")

    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"  NaN in X: {torch.isnan(X).any().item()} | "
          f"NaN in y: {torch.isnan(y).any().item()}")
    return X, y


def get_expert_mask(X: torch.Tensor, E: float) -> torch.Tensor:
    """Boolean mask selecting rows whose E_norm matches `E`."""
    E_norm_target = float(normalize_E(E))
    return (X[:, -1] - E_norm_target).abs() < 1e-4


def build_sample_weights(X: torch.Tensor) -> torch.Tensor:
    """Per-sample loss weights, derived from `EXPERT_WEIGHTS`.

    A no-op when all entries are 1.0; the dict is kept so an
    underperforming expert can be up-weighted without touching the
    training loop.
    """
    w = torch.ones(len(X))
    for E, weight in EXPERT_WEIGHTS.items():
        if weight != 1.0:
            mask = get_expert_mask(X, E)
            w[mask] = weight
    return w


def weighted_mse(pred: torch.Tensor,
                 target: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE multiplied by `weights` of shape [B]."""
    mse_per_sample = ((pred - target) ** 2).mean(dim=1)   # [B]
    return (mse_per_sample * weights).mean()


def compute_per_expert_metrics(model, X_val, y_val_logit, device):
    """Compute val MSE per expert in BOTH spaces:

      - `logit`  : MSE in pre-tanh space (the training loss)
      - `action` : MSE in post-tanh space — more interpretable but
                   loses gradient information in saturation.
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
            logit_mse  = nn.functional.mse_loss(logit_pred, ym).item()

            action_pred  = torch.tanh(logit_pred)
            action_target = torch.tanh(ym)
            action_mse   = nn.functional.mse_loss(action_pred, action_target).item()

            results[E] = {"logit": logit_mse, "action": action_mse}
    return results


# TRAINING

def train_student():
    """End-to-end training loop. See module docstring for design notes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found. Run generate_dataset first.")
        return

    X_full, y_full = load_dataset(DATASET_FILE)

    obs_dim    = X_full.shape[1] - 1
    action_dim = y_full.shape[1]
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

    # Per-sample loss weights (all 1.0 with the default config).
    sample_weights = build_sample_weights(X_full)
    print(f"  Sample weights: E=5e6 → {sample_weights[(X_full[:,-1] < 0.1).nonzero(as_tuple=False).squeeze()].mean():.2f}, "
          f"altri → 1.00")

    # Train / Val split (fixed seed for reproducibility)
    n_total = len(X_full)
    n_val   = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val

    full_dataset = TensorDataset(X_full, y_full, sample_weights)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Keep the full validation tensors handy for per-expert metric
    # computation (cheaper than re-iterating the DataLoader each epoch).
    val_indices  = val_dataset.indices
    X_val_full   = X_full[val_indices]
    y_val_full   = y_full[val_indices]

    print(f"\nTrain: {n_train:,} | Val: {n_val:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model     = StudentPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = extract_architecture(model)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Architecture        : {arch}")
    print(f"Starting student training (input={obs_dim+1}, output={action_dim})...\n")

    train_losses  = []
    val_losses    = []
    per_expert_logit  = {E: [] for E in E_VALUES}
    per_expert_action = {E: [] for E in E_VALUES}

    best_val_loss    = float('inf')
    best_epoch       = 0
    early_stop_count = 0

    for epoch in range(EPOCHS):

        # Train
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            # Add Gaussian noise to OBS only
            noise    = torch.randn(batch_X.shape[0], obs_dim, device=device) * OBS_NOISE_STD
            noisy_X  = torch.cat([batch_X[:, :obs_dim] + noise,
                                  batch_X[:, obs_dim:]], dim=1)

            optimizer.zero_grad()
            pred = model(noisy_X)
            loss = weighted_mse(pred, batch_y, batch_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y, batch_w in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                pred    = model(batch_X)
                val_loss += weighted_mse(pred, batch_y, batch_w).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # Per-expert metrics
        expert_metrics = compute_per_expert_metrics(
            model, X_val_full, y_val_full, device)
        for E in E_VALUES:
            m = expert_metrics.get(E, {"logit": float('nan'), "action": float('nan')})
            per_expert_logit[E].append(m["logit"])
            per_expert_action[E].append(m["action"])

        # Best-model selection on the weighted logit val loss (the training objective).
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch + 1
            early_stop_count = 0
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss':         best_val_loss,
                'obs_dim':          obs_dim,
                'action_dim':       action_dim,
                'target_space':     'logit',
            }, MODEL_SAVE_PATH)
        else:
            early_stop_count += 1

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            exp_str = "  ".join(
                f"E{E:.0e}: logit={expert_metrics.get(E,{}).get('logit', float('nan')):.4f}"
                f" act={expert_metrics.get(E,{}).get('action', float('nan')):.4f}"
                for E in E_VALUES
            )
            print(f"Ep {epoch+1:>3d}/{EPOCHS} | "
                  f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
                  f"LR: {current_lr:.2e} | {exp_str}")

        if early_stop_count >= EARLY_STOP_PAT:
            print(f"\nEarly stop ep {epoch+1} "
                  f"(best val={best_val_loss:.6f} @ ep {best_epoch})")
            break

    print(f"\nBest model: ep {best_epoch}, val_loss_logit={best_val_loss:.6f}")
    print(f"Saved: {MODEL_SAVE_PATH}")

    # Summary
    summary = {
        "version":           "v3",
        "target_space":      "logit",
        "best_epoch":        best_epoch,
        "best_val_loss_logit": best_val_loss,
        "final_train_loss":  train_losses[-1],
        "epochs_trained":    len(train_losses),
        "obs_dim":           obs_dim,
        "action_dim":        action_dim,
        "architecture":      arch,
        "expert_weights":    {f"E_{E:.0e}": w for E, w in EXPERT_WEIGHTS.items()},
        "per_expert_final_val": {
            f"E_{E:.0e}": {
                "logit_mse":  per_expert_logit[E][-1],
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
    """Three-panel summary plot: global loss + per-expert in both spaces."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: global train / val loss (log scale) ---
    ax = axes[0]
    ax.plot(epochs, train_losses, label='Train', color='steelblue')
    ax.plot(epochs, val_losses,   label='Val',   color='tomato')
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
               label=f'Best (ep {best_epoch})')
    ax.set_title("Global loss (weighted MSE on logits)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE (logit)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')

    # --- Plot 2: per-expert val loss in LOGIT space ---
    ax = axes[1]
    for E in E_VALUES:
        ax.plot(epochs, per_expert_logit[E], label=f"E={E:.0e}", color=E_PALETTE[E])
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
    ax.set_title("Per-expert val loss (logit space)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (logit)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')

    # --- Plot 3: per-expert val loss in ACTION space (more interpretable) ---
    ax = axes[2]
    for E in E_VALUES:
        ax.plot(epochs, per_expert_action[E], label=f"E={E:.0e}", color=E_PALETTE[E])
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
    ax.set_title("Per-expert val loss (action space)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (action)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')

    plt.tight_layout()
    path = f"{PLOTS_DIR}/distillation_loss_v3.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved to {path}")


# INFERENCE HELPER

def load_student(checkpoint_path: str, device: str = "cuda") -> StudentPolicy:
    """Load a trained student from a checkpoint.

    Example
    -------
        model  = load_student("results_student_v3/student_policy.pth")
        action = model.predict_action(obs_with_E_norm)   # tanh applied
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = StudentPolicy(ckpt['obs_dim'], ckpt['action_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded student (ep {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    print(f"  Target space: {ckpt.get('target_space', 'unknown')}")
    return model


if __name__ == "__main__":
    train_student()
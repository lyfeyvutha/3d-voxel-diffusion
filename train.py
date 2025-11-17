import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# --- External integrations ---
import wandb

# --- Project modules ---
from src.config import *
from src.dataset import VoxelDataset
from src.model import UNet3D
from src.utils import get_ddpm_schedule, q_sample, extract

# -------------------------------------------------------------------------------------------------
# 1. Setup
# -------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_INTERVAL = 50
EARLY_STOP_RECON_THRESHOLD = 1e-5
DIFFUSION_RECON_SPIKE = 0.15  # More lenient - allow bigger spikes
DIFFUSION_SPIKE_PATIENCE = 50  # Much more patience
CHECKPOINT_INTERVAL = 100


def evaluate_autoencoder_recon(model, loader):
    if loader is None:
        return None, None
    model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    with torch.no_grad():
        for batch in loader:
            clean_batch = batch.to(device) * 2.0 - 1.0
            t = torch.zeros((clean_batch.shape[0],), device=device, dtype=torch.long)
            predicted_clean = model(clean_batch, t)
            total_mse += F.mse_loss(predicted_clean, clean_batch).item()
            total_l1 += F.l1_loss(predicted_clean, clean_batch).item()
    model.train()
    batches = len(loader)
    return total_mse / batches, total_l1 / batches


def evaluate_diffusion_metrics(model, loader, schedule):
    if loader is None:
        return None, None, None
    model.eval()
    total_noise = 0.0
    total_recon = 0.0
    total_recon_l1 = 0.0
    with torch.no_grad():
        for batch in loader:
            clean_batch = batch.to(device) * 2.0 - 1.0
            t = torch.randint(0, TIMESTEPS, (clean_batch.shape[0],), device=device, dtype=torch.long)
            noisy_batch, real_noise = q_sample(clean_batch, t, schedule)
            raw_output = model(noisy_batch, t)
            predicted_clean = torch.tanh(raw_output) * PREDICTION_CLAMP
            sqrt_alphas_cumprod_t = extract(schedule["sqrt_alphas_cumprod"], t, noisy_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = torch.clamp(
                extract(schedule["sqrt_one_minus_alphas_cumprod"], t, noisy_batch.shape),
                min=1e-3,
            )
            predicted_noise = (noisy_batch - sqrt_alphas_cumprod_t * predicted_clean) / sqrt_one_minus_alphas_cumprod_t

            total_noise += F.mse_loss(predicted_noise, real_noise).item()
            total_recon += F.mse_loss(predicted_clean, clean_batch).item()
            total_recon_l1 += F.l1_loss(predicted_clean, clean_batch).item()
    model.train()
    batches = len(loader)
    return total_noise / batches, total_recon / batches, total_recon_l1 / batches


def log_exception(exc: BaseException):
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, "logs", f"train_error_{timestamp}.log")
    stack = traceback.format_exc()
    with open(log_path, "w") as f:
        f.write(stack)
    print(f"\n[Error] Training aborted due to exception: {exc}")
    print(f"Traceback written to {log_path}")


def run_training() -> int:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {device}")

    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "autoencoder_learning_rate": AUTOENCODER_LEARNING_RATE,
                "diffusion_learning_rate": DIFFUSION_LEARNING_RATE,
                "autoencoder_epochs": AUTOENCODER_PHASE_EPOCHS,
                "diffusion_epochs": DIFFUSION_PHASE_EPOCHS,
                "batch_size": BATCH_SIZE,
                "timesteps": TIMESTEPS,
                "voxel_size": VOXEL_SIZE,
            },
        )
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Error initializing wandb: {e}. Running with wandb disabled.")
        wandb.init(mode="disabled")

    print("Loading building blocks...")
    schedule = get_ddpm_schedule(TIMESTEPS, device=device)

    train_dataset = VoxelDataset(DATA_TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if os.path.isdir(DATA_VAL_DIR):
        try:
            val_dataset = VoxelDataset(DATA_VAL_DIR)
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            print(f"Validation loader enabled with {len(val_dataset)} samples.")
        except FileNotFoundError:
            val_loader = None

    model = UNet3D(in_channels=1, out_channels=1).to(device)

    print(
        f"Training configured for {AUTOENCODER_PHASE_EPOCHS} autoencoder epochs "
        f"+ {DIFFUSION_PHASE_EPOCHS} diffusion epochs (total {AUTOENCODER_PHASE_EPOCHS + DIFFUSION_PHASE_EPOCHS})."
    )

    autoencoder_best_path = os.path.join(CHECKPOINT_DIR, "autoencoder_best_model.pth")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")

    # -------------------------------------------------------------------------------------------------
    # Phase 1: Autoencoder (t = 0)
    # -------------------------------------------------------------------------------------------------
    print("\n===== Phase 1: Autoencoder (t=0) =====")
    model.train()
    auto_optimizer = Adam(model.parameters(), lr=AUTOENCODER_LEARNING_RATE)
    auto_scheduler = ReduceLROnPlateau(auto_optimizer, mode="min", factor=0.5, patience=200)

    best_auto_loss = float("inf")
    for epoch in range(1, AUTOENCODER_PHASE_EPOCHS + 1):
        total_recon_loss = 0.0
        total_recon_l1 = 0.0

        for batch in train_loader:
            auto_optimizer.zero_grad()

            clean_batch = batch.to(device) * 2.0 - 1.0
            t = torch.zeros((clean_batch.shape[0],), device=device, dtype=torch.long)
            predicted_clean = model(clean_batch, t)

            recon_loss = F.mse_loss(predicted_clean, clean_batch)
            recon_l1 = F.l1_loss(predicted_clean, clean_batch)
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            auto_optimizer.step()

            total_recon_loss += recon_loss.item()
            total_recon_l1 += recon_l1.item()

        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_recon_l1 = total_recon_l1 / len(train_loader)
        auto_scheduler.step(avg_recon_loss)

        val_ae_recon_loss, val_ae_recon_l1 = evaluate_autoencoder_recon(model, val_loader)

        if (epoch % LOG_INTERVAL == 0) or (epoch == 1):
            msg = (
                f"[AE] Epoch {epoch:04d}/{AUTOENCODER_PHASE_EPOCHS} | Recon MSE: {avg_recon_loss:.6f} | "
                f"Recon L1: {avg_recon_l1:.6f}"
            )
            if val_ae_recon_loss is not None:
                msg += f" | Val Recon MSE: {val_ae_recon_loss:.6f}"
            print(msg)

        wandb.log(
            {
                "phase": "autoencoder",
                "ae_epoch": epoch,
                "ae_train_recon_mse": avg_recon_loss,
                "ae_train_recon_l1": avg_recon_l1,
                "ae_val_recon_mse": val_ae_recon_loss,
                "ae_val_recon_l1": val_ae_recon_l1,
                "ae_total_loss": avg_recon_loss,
                "learning_rate": auto_optimizer.param_groups[0]["lr"],
            }
        )

        if avg_recon_loss < best_auto_loss:
            best_auto_loss = avg_recon_loss
            torch.save(model.state_dict(), autoencoder_best_path)
            wandb.run.summary["best_auto_loss"] = best_auto_loss

        if (
            avg_recon_loss < EARLY_STOP_RECON_THRESHOLD
            and (val_ae_recon_loss is None or val_ae_recon_loss < EARLY_STOP_RECON_THRESHOLD)
        ):
            print(f"Autoencoder early-stop triggered at epoch {epoch}.")
            break

    print(f"Autoencoder phase complete. Best recon loss: {best_auto_loss:.6f}")

    if SKIP_DIFFUSION_PHASE or DIFFUSION_PHASE_EPOCHS == 0:
        print("Skipping diffusion phase per configuration. Saving autoencoder weights as final model.")
        torch.save(model.state_dict(), final_model_path)
        torch.save(model.state_dict(), best_model_path)
        return 0

    model.load_state_dict(torch.load(autoencoder_best_path, map_location=device))

    # -------------------------------------------------------------------------------------------------
    # Phase 2: Diffusion Fine-tuning
    # -------------------------------------------------------------------------------------------------
    print("\n===== Phase 2: Diffusion Fine-tuning =====")
    model.train()
    diff_optimizer = Adam(model.parameters(), lr=DIFFUSION_LEARNING_RATE)
    diff_scheduler = ReduceLROnPlateau(diff_optimizer, mode="min", factor=0.5, patience=300)
    diff_scaler = GradScaler(enabled=(device.type == "cuda"))

    best_diff_loss = None
    best_diff_epoch = None
    best_diff_metric_label = None
    diffusion_spike_counter = 0
    for epoch in range(1, DIFFUSION_PHASE_EPOCHS + 1):
        total_noise_loss = 0.0
        total_recon_loss = 0.0
        total_recon_l1 = 0.0
        total_combined_loss = 0.0
        ramp = min(1.0, epoch / max(1, NOISE_RAMP_EPOCHS))
        noise_weight = ramp * NOISE_LOSS_TARGET
        recon_weight = RECON_LOSS_WEIGHT

        for batch in train_loader:
            diff_optimizer.zero_grad()

            clean_batch = batch.to(device) * 2.0 - 1.0
            warmup_progress = min(1.0, epoch / max(1, TIMESTEP_WARMUP_EPOCHS))
            max_cap = max(1, int(TIMESTEPS * warmup_progress))
            t = torch.randint(0, max_cap, (clean_batch.shape[0],), device=device).long()

            with autocast(enabled=(device.type == "cuda")):
                noisy_batch, real_noise = q_sample(clean_batch, t, schedule)
                raw_output = model(noisy_batch, t)
                predicted_clean = torch.tanh(raw_output) * PREDICTION_CLAMP

                sqrt_alphas_cumprod_t = extract(schedule["sqrt_alphas_cumprod"], t, noisy_batch.shape)
                sqrt_one_minus_alphas_cumprod_t = torch.clamp(
                    extract(schedule["sqrt_one_minus_alphas_cumprod"], t, noisy_batch.shape),
                    min=1e-3,
                )
                predicted_noise = (
                    (noisy_batch - sqrt_alphas_cumprod_t * predicted_clean) / sqrt_one_minus_alphas_cumprod_t
                )

                recon_loss = F.mse_loss(predicted_clean, clean_batch)
                recon_l1 = F.l1_loss(predicted_clean, clean_batch)
                noise_loss = F.mse_loss(predicted_noise, real_noise)

                combined_loss = noise_weight * noise_loss + recon_weight * recon_loss

            diff_scaler.scale(combined_loss).backward()
            diff_scaler.unscale_(diff_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            diff_scaler.step(diff_optimizer)
            diff_scaler.update()

            total_noise_loss += noise_loss.item()
            total_recon_loss += recon_loss.item()
            total_recon_l1 += recon_l1.item()
            total_combined_loss += combined_loss.item()

        avg_noise_loss = total_noise_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_recon_l1 = total_recon_l1 / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)

        if not math.isfinite(avg_combined_loss):
            print(
                f"[Diff] Epoch {epoch:04d}: combined loss {avg_combined_loss} is not finite; skipping best-model update."
            )
        diff_scheduler_metric = avg_combined_loss

        val_diff_noise_loss, val_diff_recon_loss, val_diff_recon_l1 = evaluate_diffusion_metrics(
            model, val_loader, schedule
        )
        val_ae_recon_loss, val_ae_recon_l1 = evaluate_autoencoder_recon(model, val_loader)

        val_combined_loss = None
        if val_diff_noise_loss is not None and val_diff_recon_loss is not None:
            val_combined_loss = noise_weight * val_diff_noise_loss + recon_weight * val_diff_recon_loss
            diff_scheduler_metric = val_combined_loss

        candidate_metric = avg_recon_loss
        metric_label = "train_recon"
        if val_diff_recon_loss is not None and math.isfinite(val_diff_recon_loss):
            candidate_metric = val_diff_recon_loss
            metric_label = "val_recon"

        if math.isfinite(candidate_metric) and (best_diff_loss is None or candidate_metric < best_diff_loss):
            best_diff_loss = candidate_metric
            best_diff_epoch = epoch
            best_diff_metric_label = metric_label
            torch.save(model.state_dict(), best_model_path)
            wandb.run.summary["best_diffusion_metric"] = best_diff_loss
            wandb.run.summary["best_diffusion_metric_label"] = best_diff_metric_label
            wandb.run.summary["best_diffusion_epoch"] = best_diff_epoch
            print(
                f"Saved new best diffusion model at epoch {epoch} "
                f"({metric_label} MSE {best_diff_loss:.6f})."
            )

        diff_scheduler.step(diff_scheduler_metric)

        if (epoch % LOG_INTERVAL == 0) or (epoch == 1):
            msg = (
                f"[Diff] Epoch {epoch:04d}/{DIFFUSION_PHASE_EPOCHS} | Noise MSE: {avg_noise_loss:.6f} | "
                f"Recon MSE: {avg_recon_loss:.6f} | Recon L1: {avg_recon_l1:.6f} | Total: {avg_combined_loss:.6f}"
            )
            if val_diff_recon_loss is not None:
                msg += (
                    f" | Val Diff Recon: {val_diff_recon_loss:.6f} | Val Diff Noise: {val_diff_noise_loss:.6f}"
                )
            print(msg)

        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"diffusion_checkpoint_epoch_{epoch:04d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": diff_optimizer.state_dict(),
                    "loss": avg_combined_loss,
                },
                checkpoint_path,
            )

        wandb.log(
            {
                "phase": "diffusion",
                "diff_epoch": epoch,
                "diff_train_noise_mse": avg_noise_loss,
                "diff_train_recon_mse": avg_recon_loss,
                "diff_train_recon_l1": avg_recon_l1,
                "diff_train_total_loss": avg_combined_loss,
                "diff_val_noise_mse": val_diff_noise_loss,
                "diff_val_recon_mse": val_diff_recon_loss,
                "diff_val_total_loss": val_combined_loss,
                "diff_val_recon_l1": val_diff_recon_l1,
                "ae_val_recon_mse": val_ae_recon_loss,
                "ae_val_recon_l1": val_ae_recon_l1,
                "diff_noise_weight": noise_weight,
                "learning_rate": diff_optimizer.param_groups[0]["lr"],
            }
        )

        # Early stopping DISABLED - let it train full 6000 epochs
        # The model needs time to learn, plateaus are normal
        # check_loss = val_diff_recon_loss if val_diff_recon_loss is not None else avg_recon_loss
        # if check_loss > DIFFUSION_RECON_SPIKE and epoch > 500:
        #     diffusion_spike_counter += 1
        # else:
        #     diffusion_spike_counter = 0
        # if diffusion_spike_counter >= DIFFUSION_SPIKE_PATIENCE:
        #     print(f"Diffusion early-stop triggered: {metric_label} loss > {DIFFUSION_RECON_SPIKE} for {DIFFUSION_SPIKE_PATIENCE} epochs.")
        #     break

    print(
        "Diffusion phase complete. "
        + (
            f"Best {best_diff_metric_label} MSE: {best_diff_loss:.6f}"
            if best_diff_loss is not None
            else "No best diffusion checkpoint recorded."
        )
    )
    if best_diff_epoch is not None:
        print(f"Best diffusion checkpoint came from epoch {best_diff_epoch:04d} at {best_model_path}")
    else:
        print("Warning: no best diffusion checkpoint was saved in this run.")

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best diffusion model saved to: {best_model_path}")
    return 0


def main() -> int:
    try:
        return run_training()
    except Exception as exc:
        log_exception(exc)
        return 1
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

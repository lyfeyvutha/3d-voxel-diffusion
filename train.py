import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time

import wandb # For the dashboard
from torch.cuda.amp import autocast, GradScaler # For mixed precision

# --- Import our custom files ---
from src.config import *
from src.dataset import VoxelDataset
from src.model import UNet3D
from src.utils import get_ddpm_schedule, q_sample, extract

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- WANDB SETUP ---
try:
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "autoencoder_learning_rate": AUTOENCODER_LEARNING_RATE,
            "diffusion_learning_rate": DIFFUSION_LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "timesteps": TIMESTEPS,
            "model_architecture": "UNet3D (ResNet/GroupNorm)",
            "dataset": "1-Sample Overfit Test"
        }
    )
    print("Weights & Biases initialized.")
except Exception as e:
    print(f"Error initializing wandb: {e}. Plotting will be disabled.")
    print("Please run 'wandb login' from an interactive terminal first.")
    wandb.init(mode="disabled")  # Continue training without wandb

print("Loading building blocks...")
schedule = get_ddpm_schedule(TIMESTEPS, device=device)
dataset = VoxelDataset(root_dir=DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
model = UNet3D(in_channels=1, out_channels=1).to(device)

# --- 3. Phase Configuration ---
print(f"Training configured for {AUTOENCODER_PHASE_EPOCHS} autoencoder epochs + {DIFFUSION_PHASE_EPOCHS} diffusion epochs (total {EPOCHS}).")

# Utility paths
autoencoder_best_path = os.path.join(CHECKPOINT_DIR, "autoencoder_best_model.pth")
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# -------------------------------------------------------------------------------------------------
# Phase 1: Pure Autoencoder (t = 0)
# -------------------------------------------------------------------------------------------------
print("\n===== Phase 1: Autoencoder (t=0) =====")
model.train()
auto_optimizer = Adam(model.parameters(), lr=AUTOENCODER_LEARNING_RATE)
auto_scheduler = ReduceLROnPlateau(auto_optimizer, mode='min', factor=0.5, patience=200)

best_auto_loss = float('inf')
for epoch in range(1, AUTOENCODER_PHASE_EPOCHS + 1):
    total_recon_loss = 0.0
    total_recon_l1 = 0.0

    for batch in dataloader:
        auto_optimizer.zero_grad()

        clean_batch = batch.to(device) * 2.0 - 1.0
        t = torch.zeros((clean_batch.shape[0],), device=device, dtype=torch.long)
        noisy_batch, _ = q_sample(clean_batch, t, schedule)

        predicted_clean = model(noisy_batch, t)
        recon_loss = F.mse_loss(predicted_clean, clean_batch)
        recon_l1 = F.l1_loss(predicted_clean, clean_batch)

        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        auto_optimizer.step()

        total_recon_loss += recon_loss.item()
        total_recon_l1 += recon_l1.item()

    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_recon_l1 = total_recon_l1 / len(dataloader)
    auto_scheduler.step(avg_recon_loss)

    if (epoch % 10 == 0) or (epoch == 1):
        print(f"[AE] Epoch {epoch:04d}/{AUTOENCODER_PHASE_EPOCHS} | Recon Loss: {avg_recon_loss:.6f} | Recon L1: {avg_recon_l1:.6f}")

    wandb.log({
        "phase": "autoencoder",
        "ae_epoch": epoch,
        "recon_loss": avg_recon_loss,
        "recon_error_l1": avg_recon_l1,
        "combined_loss": avg_recon_loss,
        "noise_loss": 0.0,
        "learning_rate": auto_optimizer.param_groups[0]['lr']
    })

    if avg_recon_loss < best_auto_loss:
        best_auto_loss = avg_recon_loss
        torch.save(model.state_dict(), autoencoder_best_path)
        wandb.run.summary["best_auto_loss"] = best_auto_loss

print(f"Autoencoder phase complete. Best recon loss: {best_auto_loss:.6f}")

if getattr(globals(), "SKIP_DIFFUSION_PHASE", False) or DIFFUSION_PHASE_EPOCHS == 0:
    print("Skipping diffusion phase per configuration. Saving autoencoder weights as final model.")
    torch.save(model.state_dict(), final_model_path)
    torch.save(model.state_dict(), best_model_path)
    wandb.finish()
    exit(0)

# Ensure we start diffusion from the memorised autoencoder weights
model.load_state_dict(torch.load(autoencoder_best_path, map_location=device))

# -------------------------------------------------------------------------------------------------
# Phase 2: Diffusion Fine-tuning
# -------------------------------------------------------------------------------------------------
print("\n===== Phase 2: Diffusion Fine-tuning =====")
model.train()
diff_optimizer = Adam(model.parameters(), lr=DIFFUSION_LEARNING_RATE)
diff_scheduler = ReduceLROnPlateau(diff_optimizer, mode='min', factor=0.5, patience=300)
diff_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

best_diff_loss = float('inf')
for epoch in range(1, DIFFUSION_PHASE_EPOCHS + 1):
    total_noise_loss = 0.0
    total_recon_loss = 0.0
    total_recon_l1 = 0.0
    total_combined_loss = 0.0
    
    for batch in dataloader:
        diff_optimizer.zero_grad()
        
        clean_batch = batch.to(device) * 2.0 - 1.0
        warmup_progress = min(1.0, epoch / max(1, DIFFUSION_PHASE_EPOCHS))
        max_cap = 1 + int((TIMESTEPS - 1) * warmup_progress)
        max_timestep = max(1, min(TIMESTEPS, max_cap))
        t = torch.randint(0, max_timestep, (clean_batch.shape[0],), device=device).long()
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            noisy_batch, real_noise = q_sample(clean_batch, t, schedule)
            predicted_clean = model(noisy_batch, t)
            
            sqrt_alphas_cumprod_t = extract(schedule["sqrt_alphas_cumprod"], t, noisy_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = torch.clamp(extract(schedule["sqrt_one_minus_alphas_cumprod"], t, noisy_batch.shape), min=1e-3)
            predicted_noise = (noisy_batch - sqrt_alphas_cumprod_t * predicted_clean) / sqrt_one_minus_alphas_cumprod_t

            recon_loss = F.mse_loss(predicted_clean, clean_batch)
            recon_l1 = F.l1_loss(predicted_clean, clean_batch)
            noise_loss = F.mse_loss(predicted_noise, real_noise)

            ramp = min(1.0, epoch / max(1, NOISE_RAMP_EPOCHS))
            noise_weight = min(NOISE_LOSS_TARGET, ramp * NOISE_LOSS_TARGET)
            combined_loss = (noise_weight * noise_loss) + (RECON_LOSS_WEIGHT * recon_loss)

        diff_scaler.scale(combined_loss).backward()
        diff_scaler.unscale_(diff_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        diff_scaler.step(diff_optimizer)
        diff_scaler.update()
        
        total_noise_loss += noise_loss.item()
        total_recon_loss += recon_loss.item()
        total_recon_l1 += recon_l1.item()
        total_combined_loss += combined_loss.item()

    avg_noise_loss = total_noise_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_recon_l1 = total_recon_l1 / len(dataloader)
    avg_combined_loss = total_combined_loss / len(dataloader)

    diff_scheduler.step(avg_combined_loss)

    if (epoch % 10 == 0) or (epoch == 1):
        print(
            f"[Diff] Epoch {epoch:04d}/{DIFFUSION_PHASE_EPOCHS} | Noise Loss: {avg_noise_loss:.6f} | "
            f"Recon Loss: {avg_recon_loss:.6f} | Recon L1: {avg_recon_l1:.6f} | Total Loss: {avg_combined_loss:.6f}"
        )

    wandb.log({
        "phase": "diffusion",
        "diff_epoch": epoch,
        "noise_loss": avg_noise_loss,
        "recon_loss": avg_recon_loss,
        "recon_error_l1": avg_recon_l1,
        "combined_loss": avg_combined_loss,
        "noise_weight": noise_weight,
        "learning_rate": diff_optimizer.param_groups[0]['lr']
    })
    
    if avg_combined_loss < best_diff_loss:
        best_diff_loss = avg_combined_loss
        torch.save(model.state_dict(), best_model_path)
        wandb.run.summary["best_diffusion_loss"] = best_diff_loss

print(f"Diffusion phase complete. Best combined loss: {best_diff_loss:.6f}")

torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to: {final_model_path}")
print(f"Best diffusion model saved to: {best_model_path}")
wandb.finish()

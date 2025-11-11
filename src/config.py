import os

# --- 1. WANDB (Weights & Biases) ---
WANDB_PROJECT = "3d-voxel-diffusion-v1"
WANDB_ENTITY = "lyfeyvutha"

# --- 2. File Paths ---
# Use absolute paths for Slurm
USER = os.environ.get("USER", "cvutha") 
BASE_DIR = f"/users/{USER}/Desktop/3d-voxel-diffusion"
DATA_DIR = os.path.join(BASE_DIR, "data/debug_1_sample")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

# --- 3. Data Parameters ---
VOXEL_SIZE = 64
IMG_SIZE = 64

# --- 4. Training Parameters ---
AUTOENCODER_PHASE_EPOCHS = 4000
DIFFUSION_PHASE_EPOCHS = 3000
EPOCHS = AUTOENCODER_PHASE_EPOCHS + DIFFUSION_PHASE_EPOCHS
AUTOENCODER_LEARNING_RATE = 1e-4
DIFFUSION_LEARNING_RATE = 1e-6
BATCH_SIZE = 1
NOISE_RAMP_EPOCHS = 3500

# --- 5. DDPM Algorithm Parameters ---
TIMESTEPS = 3
SAMPLING_TIMESTEPS = 3

# --- 6. Training Loss Weights ---
NOISE_LOSS_TARGET = 0.1
RECON_LOSS_WEIGHT = 20.0

SKIP_DIFFUSION_PHASE = True

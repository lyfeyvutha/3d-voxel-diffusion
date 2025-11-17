import os

# --- 1. WANDB (Weights & Biases) ---
WANDB_PROJECT = "khmer-diffusion-v1"
WANDB_ENTITY = "lyfeyvutha-brown-university"

# --- 2. File Paths ---
# Use absolute paths for Slurm
USER = os.environ.get("USER", "cvutha") # Get username from environment
BASE_DIR = f"/users/{USER}/Desktop/3d-mnist"
DATA_ROOT = os.path.join(BASE_DIR, "data", "khmer1")
DATA_TRAIN_DIR = os.path.join(DATA_ROOT, "train")
DATA_VAL_DIR = os.path.join(DATA_ROOT, "val")
DATA_DIR = DATA_TRAIN_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

# --- 3. Data Parameters ---
VOXEL_SIZE = 64
IMG_SIZE = 64

# --- 4. Training Parameters ---
TRAIN_SAMPLES_PER_NUMERAL = 200
VAL_SAMPLES_PER_NUMERAL = 40
AUTOENCODER_PHASE_EPOCHS = 50  # More epochs for better baseline
DIFFUSION_PHASE_EPOCHS = 3000  # Longer training with larger dataset
EPOCHS = AUTOENCODER_PHASE_EPOCHS + DIFFUSION_PHASE_EPOCHS
AUTOENCODER_LEARNING_RATE = 3e-4
DIFFUSION_LEARNING_RATE = 1e-4  # Higher LR - model needs to learn faster
BATCH_SIZE = 8  # Larger batch for more stable gradients with more data
NOISE_RAMP_EPOCHS = 500  # Faster ramp - introduce noise earlier

# --- 5. DDPM Algorithm Parameters ---
TIMESTEPS = 50  # More timesteps for smoother denoising
SAMPLING_TIMESTEPS = 50

# --- 6. Training Loss Weights ---
NOISE_LOSS_TARGET = 0.5  # Balanced - need to learn denoising
RECON_LOSS_WEIGHT = 1.0  # Equal weight - let model balance both
GRAD_CLIP_NORM = 0.5  # Tighter clipping for stability
PREDICTION_CLAMP = 0.95  # Slight clamping to prevent extreme values
TIMESTEP_WARMUP_EPOCHS = 500  # Matches noise ramp - gradual introduction

SKIP_DIFFUSION_PHASE = False

import torch
import torch.nn.functional as F
import math
import time
from src.config import PREDICTION_CLAMP

# Noise Schedules
def linear_beta_schedule(timesteps, device=None):
    """ DDPM noise plan that defines how much noise (β) 
    to add at each timestep (T) on the journey from T=1 to T=1000. 
    """
    beta_start = 0.0001         # This is β₁
    beta_end = 0.02             # This is β_T
    #timesteps = 1000           # This is T

    return torch.linspace(beta_start, beta_end, timesteps, device=device)

def cosine_beta_schedule(timesteps, s=0.008, device=None):
    """ Improved DDPM schedule """
    steps = timesteps + 1       # we need (1001) points to define 1000 steps. 
    # This gives us, [0., 1., 2., ..., 1000.]
    time_points = torch.linspace(0, timesteps, steps, device=device)

    # Equation (17), raw cosine function (alpha bar) or f(t)
    alphas_cumprod = torch.cos((time_points / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    # Divide the entire list by its first value to normalize 
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]                     # alpha bar = f(t)/f(0)
    # Divide the value by the one that came before it
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]    

    # Clips value between [0, 1], prevents beta value = 1.0000001 or -0.0000001
    return torch.clip(betas, 0, 0.9999)
    
def get_ddpm_schedule(timesteps, schedule_type="cosine", device=None):
    """ Calls the schedule, and precalculates all the constant tensors
        (alphas, betas, etc.) that the DDPM algorithm needs.
    """
    if schedule_type == "linear":
        betas = linear_beta_schedule(timesteps, device=device)
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(timesteps, device=device)
    else: 
        raise ValueError(f"Unknown schedule_type: {schedule_type} ")
    
    # Pre-calculate all the alphas based on the betas
    alphas = 1. - betas                                                     # alpha_t = 1 - beta_t
    alphas_cumprod = torch.cumprod(alphas, axis = 0)                        # alpha_bar, Eq. 4
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)     # alpha_{bar - 1}, Eq. 7

    # For q_sample (Forward process, Eq. 4)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                        # signal rate
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)         # standard deviation

    # For p_sample (Reverse provess, Eq. 7 and Alg. 2)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # $\beta_t$ (beta) - The "Forward Noise"
    # $\sigma_t^2$ (sigma-squared) - The "Reverse Noise"
    # $\tilde{\beta}_t$ (beta-tilde) - The "Posterior Variance"

    schedule = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }
    
    return schedule

# --- 3. HELPER FUNCTION (This is needed by q_sample and p_sample) ---

def extract(tensor, t, x_shape):
    """
    A helper function to get the correct 't' index from a tensor
    and reshape it to be broadcastable with a 5D batch. 
    Solves data shape mismatch
    """
    batch_size = t.shape[0]
    # 'gather' is a fast way to get the values from 'tensor'
    # at the indices specified by 't'
    out = tensor.gather(-1, t) 
    # Reshape to (B, 1, 1, 1, 1) so it can multiply the (B, C, D, H, W) batch
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# --- 4. DDPM ALGORITHMS ---

def q_sample(original_samples, t, schedule):
    """
    The "Forward Process" (Algorithm 1, Line 4 & 5).
    
    Why: This function implements Equation (4) from the DDPM paper:
    q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)x_0, (1-ᾱ_t)I)
    
    This function creates the "test question" (noisy_samples) and the
    "answer key" (noise) for our U-Net.
    """
    # Get the device from the input tensor itself
    device = original_samples.device
    
    # Create the "Answer Key" noise (ε)
    noise = torch.randn_like(original_samples, device=device)
    
    # Get the pre-calculated coefficients ("Rates") for this timestep 't'
    sqrt_alphas_cumprod_t = extract(schedule["sqrt_alphas_cumprod"], t, original_samples.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule["sqrt_one_minus_alphas_cumprod"], t, original_samples.shape)
    
    # This is Equation (4) implemented in code:
    # x_t = (Signal Rate * x_0) + (Noise Rate * ε)
    noisy_samples = (sqrt_alphas_cumprod_t * original_samples) + \
                    (sqrt_one_minus_alphas_cumprod_t * noise)
    
    return noisy_samples, noise

@torch.no_grad() # MUST turn off gradients for inference!
def p_sample(model, x, t, i, schedule):
    """
    One step of the "Reverse Process" (Algorithm 2, Line 4).
    This is the "chisel stroke."
    
    Why: This function implements the "magic" formula that
    denoises the shape by one step, from 't' to 't-1'.
    
    x_{t-1} = (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
    """
    # 1. Get all the pre-calculated coefficients from our "toolbox"
    betas_t = extract(schedule["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule["sqrt_one_minus_alphas_cumprod"], t, x.shape)
    alphas_t = extract(schedule["alphas"], t, x.shape)
    posterior_variance_t = extract(schedule["posterior_variance"], t, x.shape)

    # 2. Predict the clean voxel grid and derive the noise (ε_θ)
    predicted_raw = model(x, t)
    predicted_clean = torch.tanh(predicted_raw) * PREDICTION_CLAMP
    predicted_noise = (x - extract(schedule["sqrt_alphas_cumprod"], t, x.shape) * predicted_clean) / torch.clamp(extract(schedule["sqrt_one_minus_alphas_cumprod"], t, x.shape), min=1e-3)
    
    # 3. Calculate the "mean" (This is the first half of the big formula)
    #    (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * predicted_noise)
    term1 = 1. / torch.sqrt(alphas_t)
    term2 = (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise
    mean_pred = term1 * (x - term2)
    
    # 4. Add the new noise (This is the second half: + σ_t * z)
    if i == 0:
        return mean_pred # No noise on the final step (t=0)
    else:
        noise = torch.randn_like(x, device=x.device) # z ~ N(0, I)
        return mean_pred + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad() # MUST turn off gradients for inference!
def p_sample_loop(model, shape, timesteps, schedule):
    """
    The full "Sampling" loop (Algorithm 2).
    This is the "art studio" function that creates a new shape from scratch.
    
    Why: This is the user-facing "inference" function. It creates a new shape
    from pure noise by calling `p_sample` (the single-step function)
    'timesteps' times.
    """
    print("\n--- Starting Inference (Sampling) ---")
    model.eval() # Put model in evaluation mode
    
    # Get the device from the model's parameters
    device = next(model.parameters()).device
    
    # 1. Start with pure random noise (Paper, Alg. 2, Line 1)
    img = torch.randn(shape, device=device)
    
    start_time = time.time()
    
    # 2. Loop backwards from t=T...1 (Paper, Alg. 2, Line 2)
    for i in reversed(range(0, timesteps)):
        # Create a tensor for 't' (e.g., [199], [198], ...)
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        # 3. Call the single-step denoiser (Paper, Alg. 2, Line 4)
        img = p_sample(model, img, t, i, schedule)
            
    model.train() # Put model back in training mode
    end_time = time.time()
    print(f"--- Inference complete in {end_time - start_time:.2f} seconds ---")
    
    # 4. Return the final clean shape (Paper, Alg. 2, Line 6)
    return img.squeeze().cpu().numpy()

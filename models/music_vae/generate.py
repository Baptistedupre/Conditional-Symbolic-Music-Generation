import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import pretty_midi
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

# Import the VAE and denoiser components
from models.music_vae.model import MusicVAE
from models.music_vae.denoiser import DenoiseNN, sample, prepare_diffusion_schedule

# --- Preprocessing helpers (from preprocess.py) ---
def get_genres(path):
    ids = []
    genres = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != "#":
                [x, y, *_] = line.strip().split("\t")
                ids.append(x)
                genres.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
    return genre_df

def one_hot(label, num_classes):
    return np.eye(num_classes)[label].astype(int)

# -----------------------------
# Command-line Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Generate samples from VAE + Diffusion using genre prompts")
parser.add_argument("--vae_cp", type=str, default="best", choices=["best", "final"],
                    help="Which VAE checkpoint to load: 'best' or 'final'")
parser.add_argument("--diffusion_cp", type=str, default="best", choices=["best", "final"],
                    help="Which Diffusion checkpoint to load: 'best' or 'final'")
parser.add_argument("--prompts", type=str, required=True,
                    help="Comma-separated list of genre prompts (e.g., 'rock,pop,jazz')")
parser.add_argument("--genre_path", type=str, default="data/msd_tagtraum_cd1.cls",
                    help="Path to the genre label file")
parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension (should match training)")
parser.add_argument("--timesteps_diff", type=int, default=1000, help="Number of timesteps for diffusion process")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
parser.add_argument("--output_dir", type=str, default="generated_samples", help="Directory to save generated samples")
args = parser.parse_args()

# -----------------------------
# Load the Genre Mapping
# -----------------------------
genre_df = get_genres(args.genre_path)
# Get the list of unique genres from the file (order as they appear)
unique_genres = genre_df["Genre"].unique().tolist()
genre_dict = {genre: idx for idx, genre in enumerate(unique_genres)}
num_classes = len(unique_genres)
print("Available genres:", unique_genres)

# -----------------------------
# Process Prompts
# -----------------------------
prompt_list = [prompt.strip() for prompt in args.prompts.split(",")]
valid_prompts = []
conditions = []
for prompt in prompt_list:
    if prompt not in genre_dict:
        print(f"Warning: '{prompt}' is not in the available genres; skipping.")
    else:
        valid_prompts.append(prompt)
        # one_hot returns (num_classes,) vector; convert to torch tensor
        cond = torch.tensor(one_hot(genre_dict[prompt], num_classes), dtype=torch.float32)
        conditions.append(cond)
if not conditions:
    print("No valid prompts provided. Exiting.")
    sys.exit(1)

# Stack conditions for batch generation.
# Shape: (n_prompts, num_classes)
condition_tensor = torch.stack(conditions, dim=0)

# -----------------------------
# Load VAE and Diffusion Checkpoints
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up VAE architecture (use same parameters as training)
# (Assuming MusicVAE has a decode() method)
vae_model = MusicVAE(input_size=90, output_size=90, latent_dim=args.latent_dim, device=device)
vae_cp_file = os.path.join("output", f"vae_model_checkpoint_{args.vae_cp}.pt.tar")
if os.path.exists(vae_cp_file):
    print(f"Loading VAE checkpoint from {vae_cp_file}")
    cp = torch.load(vae_cp_file, map_location=device)
    vae_model.load_state_dict(cp["model_state_dict"])
else:
    print(f"VAE checkpoint not found at {vae_cp_file}. Exiting.")
    sys.exit(1)
vae_model.to(device)
vae_model.eval()

# Set up Diffusion denoiser
# (Assuming DenoiseNN parameters as in training; n_cond is set to match the condition vector size)
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=1024, n_layers=4, n_cond=num_classes, d_cond=128)
diff_cp_file = os.path.join("output", f"diffusion_model_checkpoint_{args.diffusion_cp}.pt.tar")
if os.path.exists(diff_cp_file):
    print(f"Loading Diffusion checkpoint from {diff_cp_file}")
    cp_diff = torch.load(diff_cp_file, map_location=device)
    denoise_model.load_state_dict(cp_diff["denoise_model_state_dict"])
else:
    print(f"Diffusion checkpoint not found at {diff_cp_file}. Exiting.")
    sys.exit(1)
denoise_model.to(device)
denoise_model.eval()

# Set up diffusion schedule
betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = prepare_diffusion_schedule(args.timesteps_diff)
betas = betas.to(device)

# -----------------------------
# Generate Samples
# -----------------------------
# Generate latent samples using the diffusion sampler.
# The sample() function (imported from denoiser) expects:
#   - model: the denoise_model
#   - cond: the condition tensor (shape: (batch_size, n_cond))
#   - latent_dim, timesteps, betas, batch_size
print("Generating samples for prompts:", valid_prompts)
with torch.no_grad():
    # condition_tensor shape: (n_prompts, num_classes)
    latent_samples = sample(denoise_model, condition_tensor.to(device), args.latent_dim, args.timesteps_diff, betas, batch_size=condition_tensor.size(0))
    # Use the last sample from the diffusion chain as the latent representation.
    latent = latent_samples[-1]
    
    # Decode the latent representation with the VAE decoder.
    # (Assuming MusicVAE has a decode() method.)
    # The output might be a reconstruction tensor (e.g., of shape [batch, seq_length, input_dim])
    generated_output = vae_model.decode(latent)
    
    # Move to CPU and convert to numpy for saving.
    generated_output = generated_output.cpu().numpy()

# -----------------------------
# Save Generated Samples
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)
for i, prompt in enumerate(valid_prompts):
    sample_out_path = os.path.join(args.output_dir, f"generated_{prompt}.npy")
    # Save each generated sample as a numpy array.
    np.save(sample_out_path, generated_output[i])
    print(f"Saved sample for '{prompt}' at {sample_out_path}")
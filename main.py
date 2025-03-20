import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Import VAE and its loss
from models.music_vae.model import MusicVAE
from models.music_vae.loss import ELBO_Loss
from data_processing.dataloader import MIDIDataset

# Import denoiser components
from models.music_vae.denoiser import p_losses, DenoiseNN, extract

# -----------------------------
# Command-line Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Train Music VAE and Diffusion Denoiser")
parser.add_argument("--num_epochs_vae", type=int, default=50, help="Number of epochs for VAE training")
parser.add_argument("--num_epochs_diffusion", type=int, default=20, help="Number of epochs for diffusion training")
parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension of VAE")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr_vae", type=float, default=1e-3, help="Learning rate for VAE")
parser.add_argument("--lr_diff", type=float, default=1e-3, help="Learning rate for diffusion denoiser")
parser.add_argument("--hidden_dim_diff", type=int, default=1024, help="Hidden dimension for diffusion denoiser")
parser.add_argument("--d_cond_diff", type=int, default=128, help="Conditioning d_cond for diffusion denoiser")
parser.add_argument("--n_layers_diff", type=int, default=4, help="Number of layers in diffusion denoiser")
parser.add_argument("--timesteps_diff", type=int, default=1000, help="Number of timesteps for diffusion process")
parser.add_argument("--checkpoint_interval", type=float, default=0.2, help="Fraction of total epochs after which a checkpoint is saved")
parser.add_argument("--train_from_cp_vae", action="store_true", help="Continue VAE training from checkpoint")
parser.add_argument("--train_from_cp_diffusion", action="store_true", help="Continue diffusion training from checkpoint")
args = parser.parse_args()

# -----------------------------
# Synthetic Dataset
# -----------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_length, input_dim, num_features=13):
        self.data = torch.rand(num_samples, seq_length, input_dim)
        self.features = torch.rand(num_samples, num_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # To work with both VAE and diffusion training we return a dict.
        sample = {"tensor": self.data[idx], "feature": self.features[idx]}
        # p_losses expects a "stats" attribute, so attach features.
        sample["stats"] = self.features[idx]
        return sample

# -----------------------------
# VAE Training Routine (with periodic checkpointing)
# -----------------------------
def train_vae(model, dataloader, optimizer, device, num_epochs=50, checkpoint_interval=0.2):
    model.to(device)
    best_loss = float('inf')
    best_state = None
    # Calculate number of epochs between checkpoints
    interval = max(1, int(num_epochs * checkpoint_interval))
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"VAE Epoch {epoch}"):
            inputs = batch['tensor'].to(device)
            features = batch['feature'].to(device)
            optimizer.zero_grad()
            outputs, mu, sigma, _ = model(inputs, features)
            loss = -ELBO_Loss(outputs, mu, sigma, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f"[VAE] Epoch [{epoch}/{num_epochs}] - Loss: {average_loss:.4f}")
        
        # Save best checkpoint if improvement observed.
        if average_loss < best_loss:
            best_loss = average_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "best_loss": best_loss,
                "epoch": epoch,
            }, os.path.join("output", f"vae_model_checkpoint_best.pt.tar"))
        
        # Save final checkpoint every interval and at the final epoch.
        if epoch % interval == 0 or epoch == num_epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": average_loss,
            }, os.path.join("output", f"vae_model_checkpoint_final.pt.tar"))
            
    return model, best_state, best_loss

# -----------------------------
# Setup Diffusion Hyperparameters
# -----------------------------
def prepare_diffusion_schedule(timesteps):
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# -----------------------------
# Diffusion (denoiser) Training Routine (with periodic checkpointing)
# -----------------------------
def train_diffusion(vae_model, denoise_model, dataloader, optimizer_diff, device, timesteps=1000, num_epochs=20, checkpoint_interval=0.2):
    vae_model.to(device)
    denoise_model.to(device)
    betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = prepare_diffusion_schedule(timesteps)
    
    # Move diffusion schedule tensors to device
    betas = betas.to(device)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

    best_loss = float('inf')
    best_state = None
    interval = max(1, int(num_epochs * checkpoint_interval))
    
    for epoch in range(1, num_epochs + 1):
        denoise_model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Diffusion Epoch {epoch}"):
            inputs = batch['tensor'].to(device)
            features = batch['feature'].to(device)
            # Use VAE's encoder to obtain latent representation using mu
            _, mu, _, _ = vae_model(inputs, features)
            x_start = mu

            b = x_start.shape[0]
            t = torch.randint(0, timesteps, (b,), device=device, dtype=torch.long)

            optimizer_diff.zero_grad()
            loss = p_losses(
                denoise_model, 
                x_start, 
                t, 
                features,  # Pass features directly as the condition
                sqrt_alphas_cumprod, 
                sqrt_one_minus_alphas_cumprod,
                constrain_decoder=None,
                autoencoder=None,
                loss_type="l1"
            )
            loss.backward()
            optimizer_diff.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f"[Diffusion] Epoch [{epoch}/{num_epochs}] - Loss: {average_loss:.4f}")
        
        # Save best checkpoint if improvement observed.
        if average_loss < best_loss:
            best_loss = average_loss
            best_state = {k: v.cpu().clone() for k, v in denoise_model.state_dict().items()}
            torch.save({
                "denoise_model_state_dict": best_state,
                "best_loss": best_loss,
                "epoch": epoch,
            }, os.path.join("output", f"diffusion_model_checkpoint_best.pt.tar"))
        
        # Save the final checkpoint every interval and at the final epoch.
        if epoch % interval == 0 or epoch == num_epochs:
            torch.save({
                "denoise_model_state_dict": denoise_model.state_dict(),
                "epoch": epoch,
                "loss": average_loss,
            }, os.path.join("output", f"diffusion_model_checkpoint_final.pt.tar"))
            
    return denoise_model, best_state, best_loss

# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    # Hyperparameters & settings from parser
    num_epochs_vae = args.num_epochs_vae
    num_epochs_diffusion = args.num_epochs_diffusion
    input_dim = 90
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    seq_length = 32
    feature_type = "OneHotGenre"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset setup
    if os.path.exists("data/songs/"):
        dataset = MIDIDataset("data/songs/", feature_type=feature_type)
    else:
        print("No dataset found, using synthetic dataset.")
        num_samples = 1000
        dataset = SyntheticDataset(num_samples, seq_length, input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs("output", exist_ok=True)

    # -----------------------------
    # 1. Train the VAE
    # -----------------------------
    vae_model = MusicVAE(input_size=input_dim, output_size=input_dim, latent_dim=latent_dim, device=device)
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.lr_vae)

    if args.train_from_cp_vae:
        cp_path = os.path.join("output", "vae_model_checkpoint_final.pt.tar")
        if os.path.exists(cp_path):
            print("Loading VAE checkpoint...")
            cp = torch.load(cp_path, map_location=device)
            vae_model.load_state_dict(cp["model_state_dict"])
        else:
            print("No VAE checkpoint found; starting from scratch.")

    print("Training VAE...")
    vae_model, best_state_vae, best_loss_vae = train_vae(
        vae_model, dataloader, optimizer_vae, device, 
        num_epochs=num_epochs_vae, 
        checkpoint_interval=args.checkpoint_interval)
        
    
    
    # -----------------------------
    # 2. Train the Diffusion Denoiser
    # -----------------------------
    denoise_model = DenoiseNN(
        input_dim=latent_dim,
        hidden_dim=args.hidden_dim_diff,
        n_layers=args.n_layers_diff,
        n_cond=13,
        d_cond=args.d_cond_diff
    )
    optimizer_diff = torch.optim.Adam(denoise_model.parameters(), lr=args.lr_diff)

    if args.train_from_cp_diffusion:
        cp_path_diff = os.path.join("output", "diffusion_model_checkpoint_final.pt.tar")
        if os.path.exists(cp_path_diff):
            print("Loading Diffusion checkpoint...")
            cp_diff = torch.load(cp_path_diff, map_location=device)
            denoise_model.load_state_dict(cp_diff["denoise_model_state_dict"])
        else:
            print("No Diffusion checkpoint found; starting from scratch.")

    print("Training Diffusion Denoiser...")
    denoise_model, best_state_diff, best_loss_diff = train_diffusion(
        vae_model, 
        denoise_model, 
        dataloader, 
        optimizer_diff, 
        device, 
        timesteps=args.timesteps_diff, 
        num_epochs=num_epochs_diffusion,
        checkpoint_interval=args.checkpoint_interval)
        
    
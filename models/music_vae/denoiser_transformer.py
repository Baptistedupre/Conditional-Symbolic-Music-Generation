import argparse
import math
import os
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # <--- Imported tqdm

# -----------------------------
# Sub‑modules
# -----------------------------
class NoiseEncoding(nn.Module):
    """
    Sinusoidal noise encoding.
    Given a noise tensor of shape (batch, 1), returns a tensor of shape (batch, channels).
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, noise):
        noise = noise.squeeze(-1)  # (batch,)
        assert noise.dim() == 1, f"Expected noise of dim 1, got {noise.dim()}"
        half_dim = self.channels // 2
        emb_factor = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=noise.device) * emb_factor)
        emb = 5000.0 * noise.unsqueeze(1) * emb.unsqueeze(0)
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        if self.channels % 2 == 1:
            pos_emb = F.pad(pos_emb, (0, 1))
        return pos_emb

class DenseFiLM(nn.Module):
    """
    Feature‑wise linear modulation (FiLM) generator.
    Takes a time tensor (batch,) and outputs scale and shift of shape (batch, 1, out_channels) if sequence=True.
    """
    def __init__(self, embedding_channels, out_channels, sequence=False):
        super().__init__()
        self.embedding_channels = embedding_channels
        self.out_channels = out_channels
        self.sequence = sequence
        self.noise_enc = NoiseEncoding(embedding_channels)
        self.fc1 = nn.Linear(embedding_channels, embedding_channels * 4)
        self.fc2 = nn.Linear(embedding_channels * 4, embedding_channels * 4)
        self.scale_layer = nn.Linear(embedding_channels * 4, out_channels)
        self.shift_layer = nn.Linear(embedding_channels * 4, out_channels)

    def forward(self, position):
        pos_enc = self.noise_enc(position.unsqueeze(-1))
        x = self.fc1(pos_enc)
        x = x * torch.sigmoid(x)  # swish activation
        x = self.fc2(x)
        if self.sequence:
            x = x.unsqueeze(1)
        scale = self.scale_layer(x)
        shift = self.shift_layer(x)
        return scale, shift

class DenseResBlock(nn.Module):
    """
    A simple residual block for fully‑connected layers.
    """
    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.relu = nn.ReLU()

    def forward(self, x, scale=None, shift=None):
        y = self.linear(x)
        y = self.relu(y)
        if scale is not None and shift is not None:
            y = y * scale + shift
        return x + y

class TransformerPositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    Given a tensor of positions with shape (seq_len,),
    returns a tensor of shape (seq_len, embed_channels).
    """
    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels

    def forward(self, positions):
        device = positions.device
        pos = positions.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_channels, 2, device=device, dtype=torch.float32) *
                             -(math.log(10000.0) / self.embed_channels))
        pe = torch.zeros(positions.size(0), self.embed_channels, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

# -----------------------------
# TransformerDDPM Denoising Network
# -----------------------------
class TransformerDDPM(nn.Module):
    def __init__(self, num_layers=6, num_heads=8, num_mlp_layers=2, mlp_dims=2048,
                 data_channels=512, seq_len=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.mlp_dims = mlp_dims
        self.embed_channels = 128
        self.data_channels = data_channels
        self.seq_len = seq_len

        self.input_proj = nn.Linear(data_channels, self.embed_channels)
        self.pos_enc = TransformerPositionalEncoding(self.embed_channels)
        
        self.layernorm1 = nn.LayerNorm(self.embed_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=self.embed_channels, num_heads=num_heads, batch_first=True)
        self.layernorm2 = nn.LayerNorm(self.embed_channels)
        self.ffn1 = nn.Linear(self.embed_channels, mlp_dims)
        self.ffn2 = nn.Linear(mlp_dims, self.embed_channels)
        
        self.layernorm3 = nn.LayerNorm(self.embed_channels)
        self.proj_mlp = nn.Linear(self.embed_channels, mlp_dims)
        
        self.film_layers = nn.ModuleList([
            DenseFiLM(embedding_channels=128, out_channels=mlp_dims, sequence=True)
            for _ in range(num_mlp_layers)
        ])
        self.film_blocks = nn.ModuleList([
            DenseResBlock(mlp_dims)
            for _ in range(num_mlp_layers)
        ])
        self.layernorm_out = nn.LayerNorm(mlp_dims)
        self.out_proj = nn.Linear(mlp_dims, data_channels)

    def forward(self, inputs, t):
        # t is expected to be of shape (batch,) or (batch,1)
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        temb = self.pos_enc(positions)
        temb = temb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = self.input_proj(inputs)
        x = x + temb
        
        for _ in range(self.num_layers):
            shortcut = x
            x = self.layernorm1(x)
            attn_out, _ = self.self_attn(x, x, x)
            x = shortcut + attn_out
            shortcut2 = x
            x = self.layernorm2(x)
            x = self.ffn1(x)
            x = F.gelu(x)
            x = self.ffn2(x)
            x = shortcut2 + x

        x = self.layernorm3(x)
        x = self.proj_mlp(x)
        t_squeezed = t.squeeze()
        for film, block in zip(self.film_layers, self.film_blocks):
            scale, shift = film(t_squeezed)
            x = block(x, scale=scale, shift=shift)
        x = self.layernorm_out(x)
        x = self.out_proj(x)
        return x

# -----------------------------
# Synthetic Dataset and DataLoader
# -----------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len=32, data_channels=512):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.data_channels = data_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.seq_len, self.data_channels, dtype=torch.float32)
        t = torch.rand(1, dtype=torch.float32)
        return x, t

# -----------------------------
# Loss Functions
# -----------------------------
def reduce_loss(x, reduction):
    if reduction is None or reduction == "none":
        return x
    elif reduction == "sum":
        return torch.sum(x)
    elif reduction == "mean":
        return torch.mean(x)
    else:
        raise ValueError("Unsupported reduction option.")

def denoising_score_matching_loss(batch, model, sigmas, continuous_noise=False, reduction="mean"):
    # In this loss the model is trained to predict the score: -noise/sigma^2.
    device = batch.device
    batch_size = batch.size(0)
    num_sigmas = len(sigmas)
    sigmas_tensor = torch.tensor(sigmas, dtype=batch.dtype, device=device)

    labels = torch.randint(low=int(continuous_noise), high=num_sigmas, size=(batch_size,), device=device)
    if continuous_noise:
        lower = sigmas_tensor[torch.clamp(labels - 1, min=0)]
        upper = sigmas_tensor[labels]
        rand_uniform = torch.rand_like(lower)
        used_sigmas = lower + rand_uniform * (upper - lower)
    else:
        used_sigmas = sigmas_tensor[labels]

    extra_dims = [1] * (batch.dim() - 1)
    used_sigmas = used_sigmas.view(batch_size, *extra_dims)

    noise = torch.randn_like(batch) * used_sigmas
    perturbed_samples = batch + noise

    target = -1.0 / (used_sigmas ** 2) * noise

    scores = model(perturbed_samples, used_sigmas.view(batch_size))
    
    scores_flat = scores.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    loss_per_sample = 0.5 * torch.sum((scores_flat - target_flat)**2, dim=-1) * (used_sigmas.squeeze()**2)
    return reduce_loss(loss_per_sample, reduction)

def ddpm_loss(batch, model, sigmas, continuous_noise=False, reduction="mean"):
    # In ddpm loss the model is trained to predict the noise added.
    device = batch.device
    batch_size = batch.size(0)
    num_sigmas = len(sigmas)
    sigmas_tensor = torch.tensor(sigmas, dtype=batch.dtype, device=device)

    labels = torch.randint(low=int(continuous_noise), high=num_sigmas, size=(batch_size,), device=device)
    if continuous_noise:
        lower = sigmas_tensor[torch.clamp(labels - 1, min=0)]
        upper = sigmas_tensor[labels]
        rand_uniform = torch.rand_like(lower)
        used_sigmas = lower + rand_uniform * (upper - lower)
    else:
        used_sigmas = sigmas_tensor[labels]

    extra_dims = [1] * (batch.dim() - 1)
    used_sigmas = used_sigmas.view(batch_size, *extra_dims)

    noise = torch.randn_like(batch) * used_sigmas
    perturbed_samples = batch + noise

    # For ddpm loss, we aim to predict the original noise.
    target = noise
    prediction = model(perturbed_samples, used_sigmas.view(batch_size))
    
    prediction_flat = prediction.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    loss_per_sample = nn.functional.mse_loss(prediction_flat, target_flat, reduction='none').mean(dim=-1)
    return reduce_loss(loss_per_sample, reduction)

def combined_loss(batch, model, sigmas, continuous_noise=False, reduction="mean"):
    # If using both losses, weight them equally; in this case the target is the original noise.
    return 0.5 * denoising_score_matching_loss(batch, model, sigmas, continuous_noise, reduction) + \
           0.5 * ddpm_loss(batch, model, sigmas, continuous_noise, reduction)

# -----------------------------
# Sampling Functions
# -----------------------------
def sample_dsm(model, sigmas, device, num_samples=16, sample_shape=(32,512)):
    """
    Sampling when the model is trained with dsm loss.
    Note: In dsm loss, the network approximates the score: -noise/sigma^2.
    To get a noise estimate, compute: - sigma^2 * score.
    Then update: x = x - sigma * estimated_noise.
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, *sample_shape, device=device)
        # Iterate through the full 1000-step schedule (or whatever is in sigmas)
        for sigma in tqdm(reversed(sigmas)):
            sigma_tensor = torch.full((num_samples,), sigma, device=device)
            predicted_score = model(x, sigma_tensor)
            estimated_noise = - (sigma ** 2) * predicted_score
            x = x - sigma * estimated_noise
        return x

def sample_ddpm(model, sigmas, device, num_samples=16, sample_shape=(32,512)):
    """
    Sampling when the model is trained with ddpm loss.
    Here the model predicts the noise directly.
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, *sample_shape, device=device)
        # Iterate through the full 1000-step schedule (or whatever is in sigmas)
        for sigma in tqdm(reversed(sigmas)):
            sigma_tensor = torch.full((num_samples,), sigma, device=device)
            predicted_noise = model(x, sigma_tensor)
            x = x - sigma * predicted_noise
        return x

def sample_fn(model, sigmas, device, loss_type, num_samples=16, sample_shape=(32,512)):
    """
    Wrapper sampling function:
    - For 'dsm', use sample_dsm.
    - For 'ddpm' and 'both', use ddpm sampling.
    """
    if loss_type == "dsm":
        return sample_dsm(model, sigmas, device, num_samples, sample_shape)
    else:
        return sample_ddpm(model, sigmas, device, num_samples, sample_shape)

# -----------------------------
# Checkpoint Save/Load Functions
# -----------------------------
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# -----------------------------
# Logging and Plotting Functions
# -----------------------------
def log_epoch_loss(epoch, loss, log_file="output/training_log.csv"):
    os.makedirs("output", exist_ok=True)
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["epoch", "loss"])
        writer.writerow([epoch, loss])

def plot_losses(epoch_losses, fname="output/training_loss.png"):
    os.makedirs("output", exist_ok=True)
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure()
    plt.plot(epochs, epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

# -----------------------------
# Training Loop
# -----------------------------
def train_loop(model, optimizer, dataloader, sigmas, loss_type, continuous_noise, num_epochs, device):
    best_loss = float("inf")
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        batch_losses = []
        # Variables to accumulate losses if loss type is "both"
        epoch_dsm_loss = 0.0
        epoch_ddpm_loss = 0.0
        num_batches = 0

        model.train()
        # Wrap the dataloader with tqdm for a progress bar.
        for x, t in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            if loss_type == "dsm":
                loss = denoising_score_matching_loss(x, model, sigmas, continuous_noise, reduction="mean")
            elif loss_type == "ddpm":
                loss = ddpm_loss(x, model, sigmas, continuous_noise, reduction="mean")
            elif loss_type == "both":
                dsm_val = denoising_score_matching_loss(x, model, sigmas, continuous_noise, reduction="mean")
                ddpm_val = ddpm_loss(x, model, sigmas, continuous_noise, reduction="mean")
                loss = 0.5 * dsm_val + 0.5 * ddpm_val
                epoch_dsm_loss += dsm_val.item()
                epoch_ddpm_loss += ddpm_val.item()
            else:
                raise ValueError("Unknown loss type")
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            num_batches += 1

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        # If using both, print the separate and combined losses for the epoch.
        if loss_type == "both":
            avg_dsm = epoch_dsm_loss / num_batches
            avg_ddpm = epoch_ddpm_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs} - DSM Loss: {avg_dsm:.6f}, DDPM Loss: {avg_ddpm:.6f}, Combined Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.6f}")
        save_checkpoint(model, optimizer, epoch, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, "best", best_loss)
        log_epoch_loss(epoch, avg_loss)
        plot_losses(epoch_losses)
        time.sleep(0.1)
    return model

# -----------------------------
# Main Function
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, default="dsm", choices=["dsm", "ddpm", "both"],
                        help="Select which loss to use.")
    parser.add_argument("--continuous_noise", action="store_true", help="Use continuous noise conditioning.")
    args = parser.parse_args()

    # Instantiation of dataset, model, hyperparameters (outside the train loop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 5
    num_samples = 50000  # total synthetic samples
    seq_len = 32
    data_channels = 525
    lr = 1e-3

    sigma_begin = 1.0
    sigma_end = 0.01
    # Create a training noise schedule with 15 steps:
    num_training_steps = 15
    training_sigmas = np.geomspace(sigma_begin, sigma_end, num=num_training_steps).tolist()
    # Create a sampling noise schedule with 1000 steps:
    num_sampling_steps = 1000
    sampling_sigmas = np.geomspace(sigma_begin, sigma_end, num=num_sampling_steps).tolist()

    dataset = SyntheticDataset(num_samples, seq_len=seq_len, data_channels=data_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TransformerDDPM(num_layers=6, num_heads=8, num_mlp_layers=2,
                              mlp_dims=2048, data_channels=data_channels, seq_len=seq_len)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use training_sigmas (15 steps) for training.
    trained_model = train_loop(model, optimizer, dataloader, training_sigmas,
                               args.loss_type, args.continuous_noise, num_epochs, device)

    # Use sampling_sigmas (1000 steps) for iterative refinement during sampling.
    num_sampled = 16
    samples = sample_fn(trained_model, sampling_sigmas, device, args.loss_type,
                         num_samples=num_sampled, sample_shape=(seq_len, data_channels))
    print("Sampling complete. Samples shape:", samples.shape)

if __name__ == "__main__":
    main()
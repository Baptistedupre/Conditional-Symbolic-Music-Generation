import argparse
import math
import os
import time
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from data_processing.dataloader import EmbeddingDataset

# -----------------------------
# Sub‑modules
# -----------------------------
class TransformerPositionalEncoding(nn.Module):
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

class NoiseEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, noise):
        noise = noise.squeeze(-1)  # (batch,)
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

# Modified TransformerDDPM with Brief Condition Encoding
class TransformerDDPM(nn.Module):
    def __init__(self, num_layers=6, num_heads=8, num_mlp_layers=2, mlp_dims=2048,
                 data_channels=512, seq_len=32, feature_dim=13):
        super().__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.mlp_dims = mlp_dims
        self.embed_channels = 128
        self.data_channels = data_channels
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Input projection for the song parts.
        self.input_proj = nn.Linear(data_channels, self.embed_channels)
        self.pos_enc = TransformerPositionalEncoding(self.embed_channels)
        # Brief encoder for the global condition (e.g. genre)
        self.cond_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.embed_channels)
        )
        
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

    def forward(self, inputs, t, features):
        # inputs: [BS, seq_len, data_channels]
        # features: [BS, feature_dim]
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        temb = self.pos_enc(positions)  # [seq_len, embed_channels]
        temb = temb.unsqueeze(0).expand(batch_size, -1, -1)

        # Briefly encode the condition.
        cond_emb = self.cond_encoder(features)  # [BS, embed_channels]
        cond_emb = cond_emb.unsqueeze(1).expand(batch_size, seq_len, self.embed_channels)
        
        # Process latent input and add positional and condition information.
        x = self.input_proj(inputs)  # [BS, seq_len, embed_channels]
        x = x + temb + cond_emb
        
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
# Synthetic Dataset (if needed)
# -----------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len=32, data_channels=512, feature_dim=13):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.data_channels = data_channels
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a random song split into parts and a random one-hot genre vector.
        song = torch.randn(self.seq_len, self.data_channels, dtype=torch.float32)
        # Here features is a one-hot vector represented as float (e.g., 13 classes).
        genre = torch.nn.functional.one_hot(torch.randint(0, self.feature_dim, (1,)), num_classes=self.feature_dim).squeeze(0).float()
        # Also generate a random t value.
        t = torch.rand(1, dtype=torch.float32)
        return {"tensor": song, "feature": genre, "t": t}

# -----------------------------
# Training Loop
# -----------------------------
def train_loop(model, optimizer, dataloader, sigmas, loss_type, continuous_noise, num_epochs, device):
    best_loss = float("inf")
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        batch_losses = []
        print(f"\nEpoch {epoch}/{num_epochs}")
        model.train()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            # Extract items from the dictionary batch.
            inputs = batch["tensor"].to(device)    # [BS, seq_len, data_channels]
            cond = batch["feature"].to(device)       # [BS, feature_dim]
            bs = inputs.size(0)

            # Sample noise scale from our schedule.
            sigmas_tensor = torch.tensor(sigmas, dtype=inputs.dtype, device=device)
            labels = torch.randint(low=0, high=len(sigmas), size=(bs,), device=device)
            used_sigmas = sigmas_tensor[labels]
            used_sigmas_expanded = used_sigmas.view(bs, *([1] * (inputs.dim() - 1)))

            # Add noise to inputs.
            noise = torch.randn_like(inputs) * used_sigmas_expanded
            perturbed = inputs + noise
            # For DDPM loss: target is the original noise.
            # For DSM loss: target is -noise/(sigma^2)
            if loss_type == "ddpm":
                target = noise
            else:
                target = - noise / (used_sigmas_expanded ** 2)

            # Use used_sigmas as t.
            t_used = used_sigmas.view(bs, 1)
            # Forward pass.
            output = model(perturbed, t_used, cond)
            # Compute loss.
            output_flat = output.view(bs, -1)
            target_flat = target.view(bs, -1)
            if loss_type == "ddpm":
                loss = F.mse_loss(output_flat, target_flat, reduction='mean')
            else:
                loss = F.l1_loss(output_flat, target_flat, reduction='mean')

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} Loss: {avg_loss:.6f}")
        # Save checkpoint and log loss.
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, os.path.join("checkpoints", f"checkpoint_epoch{epoch}.pt"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, os.path.join("checkpoints", f"checkpoint_epoch_best.pt"))
        # Plot losses.
        os.makedirs("output", exist_ok=True)
        epochs = list(range(1, len(epoch_losses) + 1))
        plt.figure()
        plt.plot(epochs, epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join("output", "training_loss.png"))
        plt.close()
        time.sleep(0.1)
    return model

# -----------------------------
# Sampling Functions
# -----------------------------
def sample_ddpm(model, sigmas, device, cond, num_samples=16, sample_shape=(32, 525), seed=0):
    """
    Sampling using DDPM loss.
    cond: tensor of shape [num_samples, feature_dim] 
    """
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        x = torch.randn(num_samples, *sample_shape, device=device)
        for sigma in tqdm(reversed(sigmas), desc="Sampling"):
            sigma_tensor = torch.full((num_samples, 1), sigma, device=device)
            x = x - sigma * model(x, sigma_tensor, cond)
        return x

def sample_fn(model, sigmas, device, loss_type, cond, num_samples=16, sample_shape=(32, 525), seed=0):
    if loss_type == "dsm":
        return sample_ddpm(model, sigmas, device, cond, num_samples, sample_shape, seed)
    else:
        return sample_ddpm(model, sigmas, device, cond, num_samples, sample_shape, seed)

# -----------------------------
# Main Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, default="ddpm", choices=["dsm", "ddpm", "both"],
                        help="Select which loss to use.")
    parser.add_argument("--continuous_noise", action="store_true", help="Use continuous noise conditioning.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    num_epochs = 5
    num_samples = 50000
    seq_len = 32
    data_channels = 512  # Updated channels if needed.
    feature_dim = 13
    lr = 1e-3

    sigma_begin = 1.0
    sigma_end = 0.01
    # Create a training noise schedule with 15 steps:
    num_training_steps = 15
    training_sigmas = np.geomspace(sigma_begin, sigma_end, num=num_training_steps).tolist()
    # Create a sampling noise schedule with 1000 steps:
    num_sampling_steps = 1000
    sampling_sigmas = np.geomspace(sigma_begin, sigma_end, num=num_sampling_steps).tolist()

    if os.path.exists("data/songs_embeddings/"):
        print("Loading dataset from data/songs_embeddings/")
        dataset = EmbeddingDataset("data/songs_embeddings/", feature_type="OneHotGenre")
        dataloader = EmbeddingDataset.get_dataloader(dataset, batch_size=batch_size)
    else:
        print("No dataset found, creating a random synthetic dataset.")
        dataset = SyntheticDataset(num_samples, seq_len=seq_len, data_channels=data_channels, feature_dim=feature_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TransformerDDPM(num_layers=6, num_heads=8, num_mlp_layers=2,
                              mlp_dims=2048, data_channels=data_channels, seq_len=seq_len, feature_dim=feature_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trained_model = train_loop(model, optimizer, dataloader, training_sigmas,
                               args.loss_type, args.continuous_noise, num_epochs, device)

    # For sampling we need a condition for each sample; for example, use the same fixed condition.
    sample_condition = torch.zeros(16, feature_dim, device=device)  # Adjust as needed.
    num_sampled = 16
    samples = sample_fn(trained_model, sampling_sigmas, device, args.loss_type, sample_condition,
                         num_samples=num_sampled, sample_shape=(seq_len, data_channels))
    print("Sampling complete. Samples shape:", samples.shape)

if __name__ == "__main__":
    main()
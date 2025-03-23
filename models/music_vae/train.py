import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
from tqdm import tqdm
from models.music_vae.model import MusicVAE
from models.music_vae.loss import ELBO_Loss
from data_processing.dataloader import MIDIDataset
import subprocess


def optimizer_to(optimizer, device):
    """
    Moves optimizer state tensors to the given device.
    """
    for param in optimizer.state.values():
        # Each state is a dict, and might include tensors such as momentum buffers.
        for key, value in param.items():
            if isinstance(value, torch.Tensor):
                param[key] = value.to(device)


def train(
    model: MusicVAE,
    dataloader: MIDIDataset,
    optimizer: torch.optim.Adam,
    device: str = "cuda",
    num_epochs: int = 50,
    resume_point: int = 0,
):
    model.to(device)
    for epoch in range(resume_point + 1, num_epochs + 1):
        print(f" Starting epoch {epoch}")
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            inputs = batch["tensor"].to(device)
            features = batch["feature"].to(device)
            optimizer.zero_grad()
            outputs, mu, sigma, _ = model(inputs, features)
            loss = -ELBO_Loss(outputs, mu, sigma, inputs)
            loss.backward()
            # Gradient clipping to avoid exploding gradients (max norm = 1e5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e5)
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {average_loss:.4f}")

        if epoch % 1 == 0:
            import os

            os.makedirs("output", exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average_loss": average_loss,
            }
            torch.save(checkpoint, os.path.join("output", "model.pt"))

            command = [
                "mc",
                "cp",
                os.path.expanduser("~/work/MusicVAE/output/model.pt"),
                f"s3/lstepien/Conditional_Music_Generation/data/model_epoch_{epoch}.pt",
            ]

            # Execute the command.
            subprocess.run(command, check=True)
            print(f"Checkpoint for epoch {epoch} moved to S3 folder")


if __name__ == "__main__":
    num_epochs = 50
    input_dim = 90
    feature_type = "OneHotGenre"
    
    if os.path.exists("data/songs/"):
        dataset = MIDIDataset("data/songs/", feature_type=feature_type)
        dataloader = MIDIDataset.get_dataloader(dataset, batch_size=16)
    else:
        print("No dataset found, creating a random one.")

        from torch.utils.data import Dataset, DataLoader

        class SyntheticDataset(Dataset):
            def __init__(self, num_samples, seq_length, input_dim, num_features=13):
                self.data = torch.rand(num_samples, seq_length, input_dim)
                self.features = torch.rand(num_samples, num_features)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"tensor": self.data[idx], "feature": self.features[idx]}

        seq_length = 32
        num_samples = 1000
        batch_size = 16

        dataset = SyntheticDataset(num_samples, seq_length, input_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicVAE(
        input_size=input_dim, output_size=input_dim, latent_dim=512, device=device
    )  # noqa 501
    checkpoint = torch.load("output/model.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    starting_epoch = checkpoint["epoch"]
    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(optimizer_state_dict)
    optimizer_to(optimizer, device)

    ################test
    epoch = starting_epoch
    command = [
        "mc",
        "cp",
        os.path.expanduser(
            "~/work/MusicVAE/output/model.pt"
        ),  # expands the "~" properly
        f"s3/lstepien/Conditional_Music_Generation/data/model_epoch_{epoch}.pt",
    ]

    # This line will run the command:
    subprocess.run(command, check=True)
    print("test checkpoint moved to S3 folder (verify)")
    del epoch
    ##############

    train(model, dataloader, optimizer, device, num_epochs, starting_epoch)

    # Save the trained model as a pt.tar file in the output folder.
    import os

    os.makedirs("output", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_epochs": num_epochs,
    }
    torch.save(checkpoint, os.path.join("output", "model.pt"))

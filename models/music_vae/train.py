import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from tqdm import tqdm
from models.music_vae.model import MusicVAE
from models.music_vae.loss import ELBO_Loss
from data_processing.dataloader import MIDIDataset


def train(
    model: MusicVAE,
    dataloader: MIDIDataset,
    optimizer: torch.optim.Adam,
    device: str = "cuda",
    num_epochs: int = 50,
):
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            inputs = batch['tensor'].to(device)
            features = batch['feature'].to(device)
            optimizer.zero_grad()
            outputs, mu, sigma, _ = model(inputs, features)
            loss = -ELBO_Loss(outputs, mu, sigma, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {average_loss:.4f}")


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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, dataloader, optimizer, device, num_epochs)
    
    # Save the trained model as a pt.tar file in the output folder.
    import os
    os.makedirs("output", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_epochs": num_epochs,
    }
    torch.save(checkpoint, os.path.join("output", "model.pt.tar"))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from model import MusicVAE
from loss import ELBO_Loss


def train(model, dataloader, optimizer, device, num_epochs):
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs, mu, sigma, _ = model(inputs)
            loss = -ELBO_Loss(outputs, mu, sigma, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {average_loss:.4f}")


if __name__ == '__main__':
    # Exemple de dataset factice : remplacez ceci par votre DataLoader réel
    seq_length = 32
    input_dim = 90
    num_samples = 1000
    batch_size = 16
    num_epochs = 10

    # Création d'un dataset aléatoire
    data = torch.rand(num_samples, seq_length, input_dim)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # noqa 501
    model = MusicVAE(input_size=input_dim, output_size=input_dim, latent_dim=512, device=device) # noqa 501
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, dataloader, optimizer, device, num_epochs)

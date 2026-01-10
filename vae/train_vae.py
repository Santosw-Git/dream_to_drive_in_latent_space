import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vae import ConvVAE


class CarDataset(Dataset):
    def __init__(self, image_dir):
        self.files = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = self.transform(img)
        return img


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon = torch.mean((x_hat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CarDataset("dataset/images")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

vae = ConvVAE(latent_dim=32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0
    for x in loader:
        x = x.to(device)

        x_hat, mu, logvar = vae(x)
        loss = vae_loss(x_hat, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

torch.save(vae.state_dict(), "vae.pth")
print("VAE training complete.")

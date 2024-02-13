from gan import Generator
from encoder import ImageEncoder
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Trainer:
    def __init__(self, epochs=4, batch_size=8, lr=1e-4, embedding_size=32000, save_every=100):
        self.generator = Generator(latent_dim=embedding_size).to(device)
        self.encoder = ImageEncoder(latent_dim=embedding_size).to(device)
        self.optim = torch.optim.RAdam(self.encoder.parameters(), lr=lr)
        self.dataset = Dataset(max_ctx_length=0, data_aug=True)
        self.train_dataloader = D.DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
        self.step = 0
        self.save_every = save_every
        self.epochs = epochs
        self.embedding_size = embedding_size

    def train(self):
        """Train the model."""
        print("Beginning Training...")
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            bar = tqdm(enumerate(self.train_dataloader))
            for i, (x) in bar:
                if not i % self.save_every == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')
                else:
                    self.save()
                bar.update(1)

    def training_step(self, x):
        x = x.to(device)
        self.optim.zero_grad()

        # FAKE DATA
        latent = torch.randint(0, self.embedding_size, (x.shape[0],), device=device)
        fake = self.generator(torch.zeros_like(x), latent.unsqueeze(1))
        fake_encoded = self.encoder(fake)

        loss = F.cross_entropy(fake_encoded, latent)
        loss.backward()
        self.optim.step()
        return loss.item()

    def save(self):
        torch.save(self.encoder.state_dict(), 'encoder.pt')
        print("Saved model.")

if __name__ == "__main__":
    device = torch.device('mps')  # Define the device
    trainer = Trainer()
    trainer.train()
    trainer.save()
    print("Training complete.")

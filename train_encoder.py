from gan import Generator
from encoder import ImageEncoder

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
from tqdm import tqdm

class Trainer:
    def __init__(self, epochs=4,
                       batch_size=8,
                       lr=1e-4,
                       embedding_size=32000,
                       save_every=100,
    ):
        self.generator = Generator(latent_dim=embedding_size).to('mps')
        gw = torch.load('engine.pt', map_location='mps')
        self.generator.load_state_dict(gw)
        self.encoder = ImageEncoder(latent_dim=embedding_size).to('mps')
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
        self.encoder.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.encoder.train()
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if not i % self.save_every == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')
                else:
                    self.save()
                bar.update(1)

    def training_step(self, x):
        #Get a batch of images
        x = x.to('mps')
        self.optim.zero_grad()

        #FAKE DATA

        #Generate a random latent vector
        latent = torch.randint(0, self.embedding_size, (x.shape[0],), device=x.device)
        #Generate a fake image
        fake = self.generator(torch.zeros_like(x), latent.unsqueeze(1))

        #Encode the fake image
        fake_encoded = self.encoder(fake)

        #Get the loss
        loss = F.cross_entropy(fake_encoded, latent)
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def save(self):
        torch.save(self.encoder.state_dict(), 'encoder.pt')
        print("Saved model.")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.save()
    print("Training complete.")
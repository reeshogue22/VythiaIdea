import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from gan import Generator, Discriminator
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
from tqdm import tqdm

class Trainer:
    def __init__(self, epochs=4,
                       batch_size=8,
                       lr=1e-4,
                       dlr=1e-5,
                       save_every=100,
                       embedding_size=32000
            ):
        self.epochs = epochs
        self.save_every = save_every
        self.embedding_size = embedding_size
        self.engine = Generator(latent_dim=embedding_size).to('mps')
        self.discriminator = Discriminator().to('mps')
        self.optim = torch.optim.RAdam(self.engine.parameters(), lr=lr)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=dlr)
        print(get_nparams(self.engine), "params in generator net.")
        self.dataset = Dataset(max_ctx_length=0, data_aug=True)
        self.train_dataloader = D.DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
        self.step = 0

    def train(self):
        """Train the model."""

        print("Beginning Training...")
        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.engine.train()
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if not i % self.save_every == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')
                else:
                    self.save()
                bar.update(1)
    
    def training_step(self, x):
        """
        One optimization step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x = x.to('mps')
        self.optim.zero_grad()
        x_hat = self.engine(torch.zeros_like(x), torch.randint(0, self.embedding_size, (x.shape[0], 1), device=x.device))
        disc_false = self.discriminator(x_hat)
        loss = F.binary_cross_entropy_with_logits(disc_false, torch.ones_like(disc_false))
        loss.backward()
        self.optim.step()

        self.discriminator_optim.zero_grad()
        disc_true = self.discriminator(x)
        x_hat = self.engine(torch.zeros_like(x), torch.randint(0, self.embedding_size, (x.shape[0], 1), device=x.device))
        disc_false = self.discriminator(x_hat)
        dloss = F.binary_cross_entropy_with_logits(disc_true, torch.ones_like(disc_true)) + F.binary_cross_entropy_with_logits(disc_false, torch.zeros_like(disc_false))
        dloss.backward()
        self.discriminator_optim.step()

        return loss.item()

    def save(self):
        """Save the model."""
        print("Saving...")
        torch.save(self.engine.state_dict(), 'engine.pt')
        torch.save(self.discriminator.state_dict(), 'discriminator.pt')
        print("Saved.")


if __name__ == "__main__":
    trainer = Trainer()
    atexit.register(trainer.save)
    trainer.train()
    trainer.save()
    print("Training complete.")
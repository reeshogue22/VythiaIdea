import transformers
import torch
from gan import Generator
from encoder import ImageEncoder
from dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import RAdam
from tqdm import tqdm
from utils import show_tensor, init_camera_and_window
camera, display, surface = init_camera_and_window()
        
#Finetuning a lm to generate video frames instead of text
class Trainer:
    def __init__(self, epochs=4,
                       batch_size=1,
                       lr=1e-4,
                       embedding_size=32000,
                       save_every=100,
    ):
        self.generator = Generator(latent_dim=embedding_size                                ).to('mps')
        gw = torch.load('engine.pt', map_location='mps')
        self.generator.load_state_dict(gw)
        self.encoder = ImageEncoder(latent_dim=embedding_size).to('mps')
        ew  = torch.load('encoder.pt', map_location='mps')
        self.encoder.load_state_dict(ew)
        self.lm = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to('mps')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        self.optim = RAdam(self.lm.parameters(), lr=lr)
        self.dataset = Dataset(max_ctx_length=128, data_aug=True)
        self.train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
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
                    self.test_step(x)
                    self.save()
                bar.update(1)

    def training_step(self, x):
        #Encode the images
        x = x.to('mps')
        self.optim.zero_grad()
        encoded = self.encoder(x)

        #Get argmax of the encodings
        tokenized = torch.argmax(encoded, dim=1).to(torch.int)
        print(tokenized)
        print(tokenized.shape)
        lm = self.lm(tokenized, labels=tokenized)
        loss = lm.loss
        loss.backward()

        self.optim.step()
        return loss.item()

    def save(self):
        torch.save(self.lm.state_dict(), 'lm.pt')
    
    def test_step(self, x):
        x = x.to('mps')

        #Encode the images
        encoded = self.encoder(x)
        encoded = torch.argmax(encoded, dim=1).to(torch.int)
        print(encoded)
        to_lm = self.lm(encoded)
        to_lm = to_lm.logits
        to_lm = torch.argmax(to_lm, dim=-1).unsqueeze(-1)
        
        #Send to generator
        generated = self.generator(torch.zeros_like(x), to_lm)

        for i in range(generated.shape[-1]):
            show_tensor(generated[:,:,:,:,i], display, surface)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.save()
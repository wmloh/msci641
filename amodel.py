import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from utils.custom_torch import tanh_loss
from block import DownBlock, SimBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AModel(nn.Module):
    def __init__(self, dropout=0.5, num_simblocks=2, embedding_size=16):
        super().__init__()
        self.sim_blocks = nn.ModuleList([SimBlock(768) for _ in range(num_simblocks)])

        self.block_h1 = DownBlock(768, 256, 0.)
        self.block_h2 = DownBlock(256, 128, 0.)
        self.block_h3 = DownBlock(128, 64, dropout)
        self.block_h4 = DownBlock(64, 16, dropout)

        self.block_b1 = DownBlock(768, 256, 0.)
        self.block_b2 = DownBlock(256, 128, 0.)
        self.block_b3 = DownBlock(128, 64, dropout)
        self.block_b4 = DownBlock(64, 16, dropout)

        self.block1 = DownBlock(768 * 2 + 16 * 2, 256, dropout)
        self.block2 = DownBlock(256, 64, dropout)
        # self.block3 = DownBlock(64, 16, dropout)
        self.fc1 = nn.Linear(64, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 4)

    def forward(self, headline, body):
        a1 = headline
        a2 = body
        for sblock in self.sim_blocks:
            a1, a2 = sblock(a1, a2)

        h = self.block_h1(headline)
        h = self.block_h2(h)
        h = self.block_h3(h)
        h = self.block_h4(h)

        b = self.block_b1(body)
        b = self.block_b2(b)
        b = self.block_b3(b)
        b = self.block_b4(b)

        embed = torch.concat((a1, a2, h, b), 1)
        embed = self.block1(embed)
        embed = self.block2(embed)
        # embed = self.block3(embed)
        embed = torch.sigmoid(self.fc1(embed))
        embed = F.softmax(self.fc2(embed), 1)

        return embed

    def predict(self, headline, body):
        out = self(headline, body)
        return torch.argmax(out, dim=1)

    def predict_proba(self, headline, body):
        a1 = headline
        a2 = body
        for sblock in self.sim_blocks:
            a1, a2 = sblock(a1, a2)

        h = self.block_h1(headline)
        h = self.block_h2(h)
        h = self.block_h3(h)
        h = self.block_h4(h)

        b = self.block_b1(body)
        b = self.block_b2(b)
        b = self.block_b3(b)
        b = self.block_b4(b)

        embed = torch.concat((a1, a2, h, b), 1)
        embed = self.block1(embed)
        embed = self.block2(embed)
        embed = torch.sigmoid(self.fc1(embed))

        return embed

    @classmethod
    def fit(cls, model, dl, dl_val, epochs=20, optim_fn=torch.optim.Adam, loss_type=nn.CrossEntropyLoss,
            save_path=None, display=False, device=DEVICE, loss_args={}):
        optim = optim_fn(model.parameters(), lr=9e-4)
        loss_fn = loss_type(**loss_args)
        model.to(device)

        losses = list()
        val_losses = list()
        best_val_loss = torch.inf
        best_model = None
        pbar = trange(epochs)

        for e in pbar:
            curr_loss = 0
            model.train()

            for (headline, body), label in dl:
                headline = headline.float().to(device)
                body = body.float().to(device)
                label = label.type(torch.LongTensor).to(device)

                optim.zero_grad()
                pred = model(headline, body)
                loss = loss_fn(pred, label).float()
                loss.backward()
                optim.step()

                curr_loss += loss.item()

            val_loss = 0
            model.eval()
            for (headline, body), label in dl_val:
                headline = headline.float().to(device)
                body = body.float().to(device)
                label = label.type(torch.LongTensor).to(device)

                val_loss += loss_fn(model(headline, body), label).float().item()

            losses.append(curr_loss / len(dl))
            val_losses.append(val_loss / len(dl_val))
            pbar.set_description(f'Loss: {losses[-1]} | val loss: {val_losses[-1]}')

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model = model.state_dict()

        model.load_state_dict(best_model)
        model.eval()
        model.cpu()

        if save_path is not None:
            assert isinstance(save_path, str)
            torch.save(model, save_path)

        if display:
            plt.figure()
            plt.plot(losses, label='train')
            plt.plot(val_losses, label='val')
            plt.ylabel('Cross entropy loss')
            plt.xlabel('Epoch number')
            plt.legend()

        return losses, val_losses

    @classmethod
    def combine_features(cls, dl, model, X_base, y):
        model.to(DEVICE)
        pred = list()

        with torch.inference_mode():
            for (headline, body), label in dl:
                headline = headline.float().to(DEVICE)
                body = body.float().to(DEVICE)

                pred += model.predict_proba(headline, body).cpu().tolist()

        pred = np.asarray(pred)
        features = np.hstack((pred, X_base))

        model.cpu()

        return features, y

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from utils.custom_torch import tanh_loss
from block import DownBlock, SimBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AModel(nn.Module):
    def __init__(self, dropout=0.5, num_simblocks=2):
        super().__init__()
        self.sim_blocks = nn.ModuleList([SimBlock(768) for _ in range(num_simblocks)])

        self.block1 = DownBlock(768 * 2, 256, dropout)
        self.block2 = DownBlock(256, 64, dropout)
        self.block3 = DownBlock(64, 16, dropout)
        self.fc = nn.Linear(16, 1)

    def forward(self, headline, body):
        a1 = headline
        a2 = body
        for sblock in self.sim_blocks:
            a1, a2 = sblock(a1, a2)

        embed = torch.concat((a1, a2), 1)
        embed = self.block1(embed)
        embed = self.block2(embed)
        embed = self.block3(embed)
        embed = torch.tanh(self.fc(embed)).squeeze(1)

        return embed

    @classmethod
    def fit(cls, model, dl, dl_val, epochs=20, optim_fn=torch.optim.Adam, loss_type=nn.MSELoss,
            save_path=None, display=False, device=DEVICE):
        optim = optim_fn(model.parameters(), lr=5e-4)
        loss_fn = loss_type()
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
                label = label.float().to(device)

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
                label = label.float().to(device)

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
            plt.ylabel('MSE loss')
            plt.xlabel('Epoch number')
            plt.legend()

        return losses, val_losses

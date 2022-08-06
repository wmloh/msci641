import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from block import DownBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.block1 = DownBlock(768, 512, dropout)
        self.block2 = DownBlock(1280, 256, dropout)
        self.block3 = DownBlock(1536, 64, dropout)
        self.block4 = DownBlock(1600, 128, dropout)
        self.block5 = DownBlock(128, 32, dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, body):

        z1 = self.block1(body)
        z1 = torch.concat((z1, body), 1)  # 1280

        z2 = self.block2(z1)  # 256
        z2 = torch.concat((z2, z1), 1)  # 1536

        z3 = self.block3(z2)  # 64
        z3 = torch.concat((z3, z2), 1)  # 1600

        z4 = self.block4(z3)  # 128
        z5 = self.block5(z4)  # 32
        z6 = torch.sigmoid(self.fc(z5))  # 1

        return z6.squeeze(1)

    @classmethod
    def fit(cls, model, dl, dl_val, epochs=40, optim_fn=torch.optim.Adam, loss_type=nn.BCELoss,
            save_path=None, display=False, device=DEVICE):
        optim = optim_fn(model.parameters())
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
            for body, label in dl:
                body = body.float().to(device)
                label = label.float().to(device)

                optim.zero_grad()
                pred = model(body)
                loss = loss_fn(pred, label).float()
                loss.backward()
                optim.step()

                curr_loss += loss.item()

            val_loss = 0
            model.eval()
            for body, label in dl_val:
                body = body.float().to(device)
                label = label.float().to(device)

                val_loss += loss_fn(model(body), label).float().item()

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
            plt.ylabel('BCE loss')
            plt.xlabel('Epoch number')
            plt.legend()

        return losses, val_losses

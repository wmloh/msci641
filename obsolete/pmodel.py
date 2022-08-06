import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.block1 = DownBlock(768, 512, dropout)
        self.block2 = DownBlock(512, 256, dropout)
        self.block3 = DownBlock(256, 64, dropout)
        self.block4 = DownBlock(64, 16, dropout)
        self.block5 = DownBlock(16, 4, dropout)
        self.fc = nn.Linear(4, 1)

        self.fc1 = nn.Linear(768, 512)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(800, 64)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(832, 128)
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(896, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, body):
        z = F.relu(self.fc1(body))
        z = torch.concat((self.drop1(body), z), 1)

        z = F.relu(self.fc2(z))
        z = torch.concat((self.drop2(body), z), 1)

        z = F.relu(self.fc3(z))
        z = torch.concat((self.drop3(body), z), 1)

        z = F.relu(self.fc4(z))
        z = torch.sigmoid(self.fc5(z))

        return z.squeeze(1)

    @classmethod
    def fit(cls, model, dl, dl_val, epochs=40, optim_fn=torch.optim.Adam, loss_type=nn.BCELoss,
            save_path=None, display=False, device=DEVICE):
        optim = optim_fn(model.parameters())
        loss_fn = loss_type()
        model.to(device)
        model.train()

        losses = list()
        val_losses = list()
        best_val_loss = torch.inf
        best_model = None
        pbar = trange(epochs)

        for e in pbar:
            curr_loss = 0
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


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        self.fc = nn.Linear(in_dim, out_dim)
        self.batch = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop(self.batch(F.relu(self.fc(x))))

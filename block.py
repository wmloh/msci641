import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.batch = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop(self.batch(F.relu(self.fc(x))))


class SimBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.tensor(1./in_dim))
        self.sim_W1 = nn.Parameter(torch.randn(in_dim, 1), requires_grad=True)
        self.sim_W2 = nn.Parameter(torch.randn(in_dim, 1), requires_grad=True)
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.layernorm2 = nn.LayerNorm(in_dim)

    def forward(self, x, y):
        outer = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
        sim1 = torch.matmul(outer, self.sim_W1).squeeze(2) * self.scale_factor
        sim2 = torch.matmul(outer, self.sim_W2).squeeze(2) * self.scale_factor

        sim_x = F.softmax(sim1, dim=1) * x
        sim_y = F.softmax(sim2, dim=1) * y

        output_x = sim_x + x
        output_y = sim_y + y

        output_x = self.layernorm1(output_x)
        output_y = self.layernorm2(output_y)

        return output_x, output_y


class JointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)

        self.layernorm1 = nn.LayerNorm((768, 1))
        self.layernorm2 = nn.LayerNorm((768, 1))

    def forward(self, x, y):
        a1 = self.attn1(x, y, y, need_weights=False)[0]
        a2 = self.attn2(y, x, x, need_weights=False)[0]

        a1 = self.layernorm1(a1)
        a2 = self.layernorm2(a2)

        a1 = a1 + x
        a2 = a2 + y

        return a1, a2


if __name__ == '__main__':
    x = torch.randn((4, 768))
    y = torch.randn((4, 768))
    sb = SimBlock(768)

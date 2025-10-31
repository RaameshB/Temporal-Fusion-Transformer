import torch
import torch.nn as nn

class GRN(nn.Module):
    """
    Gated Residual Network
    """
    def __init__(self, primary_dim, hidden_dim, context_dim=0):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, primary_dim)
        self.dense2 = nn.Linear(primary_dim, hidden_dim)
        self.use_context = context_dim > 0
        self.dense3 = nn.Linear(context_dim, hidden_dim, bias=False) if self.use_context else None
        self.glu = nn.GLU(dim=-1)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(primary_dim)

    def forward(self, a, c=0):
        if self.use_context:
            eta2 = self.elu(self.dense2(a) + self.dense3(c))
        else:
            eta2 = self.elu(self.dense2(a))
        eta1 = self.dense1(eta2)
        eta1_reshaped = torch.unsqueeze(eta1, -1).expand((*eta1.shape, 2))
        y = self.layer_norm(a + torch.squeeze(self.glu(eta1_reshaped), -1))
        return y


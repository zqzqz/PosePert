import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttentionWithTrust(nn.Module):
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.sqrt_dim = math.sqrt(dim)
        self.temperature = temperature
        self.score = None
        self.attn  = None

    def forward(self, query, key, value, trust_score=None) -> torch.Tensor:
        self.score = torch.bmm(query, key.transpose(1,2)) / self.sqrt_dim
        if trust_score is not None:
            if trust_score.dim() == 2:
                trust = trust_score.unsqueeze(1)
            else:
                trust = trust_score
            self.score = self.score * trust_score

        self.attn = F.softmax(self.score / self.temperature, dim=-1)
        
        if trust_score is not None:
            self.attn = self.attn * trust_score
            self.attn = self.attn / (self.attn.sum(dim=-1, keepdim=True))
        context = torch.bmm(self.attn, value)
        return context

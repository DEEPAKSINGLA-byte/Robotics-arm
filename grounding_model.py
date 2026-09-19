"""A small trainable candidate selector on top of frozen SigLIP features."""
import torch
from torch import nn
import torch.nn.functional as F


class GroundingModel(nn.Module):
    def __init__(self, feature_dim=768, hidden=256, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, hidden=hidden, layers=layers, heads=heads, dropout=dropout)
        self.visual = nn.Linear(feature_dim, hidden)
        self.text = nn.Linear(feature_dim, hidden)
        self.scene = nn.Linear(feature_dim, hidden)
        self.geometry = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*4, dropout, batch_first=True, norm_first=True)
        self.relations = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, visual, text, scene, geometry, valid):
        visual = F.normalize(visual, dim=-1)
        text = F.normalize(text, dim=-1)
        scene = F.normalize(scene, dim=-1)
        candidates = self.visual(visual) + self.geometry(geometry) + self.text(text).unsqueeze(1)
        context = torch.stack([self.text(text), self.scene(scene)], dim=1)
        tokens = torch.cat([context, candidates], dim=1)
        context_valid = torch.ones((valid.shape[0], 2), dtype=torch.bool, device=valid.device)
        padding = ~torch.cat([context_valid, valid], dim=1)
        related = self.relations(tokens, src_key_padding_mask=padding)[:, 2:]
        similarity = (visual * text.unsqueeze(1)).sum(-1)
        scores = self.score(related).squeeze(-1) + self.similarity_scale * similarity
        return scores.masked_fill(~valid, float('-inf'))

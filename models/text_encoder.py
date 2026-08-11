
from typing import List

import clip
import torch
import torch.nn as nn


class DemandTextEncoder(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        token_len: int = 16,
        token_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.token_len = token_len
        self.token_dim = token_dim
        self.device = device

        self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            tok = clip.tokenize(["dummy"]).to(device)
            feat = self.clip_model.encode_text(tok).float()
            self.clip_dim = feat.shape[-1]

        self.proj = nn.Linear(self.clip_dim, token_len * token_dim)
        self.pos_emb = nn.Parameter(torch.zeros(token_len, token_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @torch.no_grad()
    def _encode_clip(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        feat = self.clip_model.encode_text(tokens).float()
        return feat

    def forward(self, texts: List[str]) -> torch.Tensor:
        e = self._encode_clip(texts)
        t = self.proj(e)
        t = t.view(-1, self.token_len, self.token_dim)
        t = t + self.pos_emb.unsqueeze(0)
        return t
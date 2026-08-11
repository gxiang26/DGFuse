
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcmm import DCMM


class TimeEmbedding(nn.Module):
    def __init__(self, sinu_dim: int = 128, out_ch: int = 4):
        super().__init__()
        self.sinu_dim = sinu_dim
        self.out_ch = out_ch
        self.mlp = nn.Sequential(
            nn.Linear(sinu_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_ch),
        )

    def forward(self, t: torch.Tensor, spatial: Tuple[int, int]) -> torch.Tensor:
        device = t.device
        half = self.sinu_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        emb = self.mlp(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        emb = emb.expand(-1, -1, spatial[0], spatial[1])
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,
        ir_ch: int = 1,
        vi_ch: int = 3,
        time_ch: int = 4,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        token_dim: int = 256,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.img_ch = img_ch
        self.ir_ch = ir_ch
        self.vi_ch = vi_ch
        self.time_ch = time_ch

        in_ch_total = img_ch + ir_ch + vi_ch + time_ch
        self.time_emb = TimeEmbedding(sinu_dim=128, out_ch=time_ch)
        self.stem = nn.Conv2d(in_ch_total, base_ch, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_dcmms = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_ch
        enc_chs = [ch]
        for i, m in enumerate(ch_mults):
            out_c = base_ch * m
            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(ch, out_c))
                self.enc_dcmms.append(DCMM(out_c, token_dim=token_dim,
                                           ir_ch=ir_ch, vi_ch=vi_ch,
                                           heads=attn_heads))
                ch = out_c
                enc_chs.append(ch)
            if i != len(ch_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                enc_chs.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid1 = ResBlock(ch, ch)
        self.mid_dcmm = DCMM(ch, token_dim=token_dim, ir_ch=ir_ch,
                             vi_ch=vi_ch, heads=attn_heads)
        self.mid2 = ResBlock(ch, ch)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_dcmms = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, m in reversed(list(enumerate(ch_mults))):
            out_c = base_ch * m
            for _ in range(num_res_blocks + 1):
                skip_c = enc_chs.pop()
                self.dec_blocks.append(ResBlock(ch + skip_c, out_c))
                self.dec_dcmms.append(DCMM(out_c, token_dim=token_dim,
                                           ir_ch=ir_ch, vi_ch=vi_ch,
                                           heads=attn_heads))
                ch = out_c
            if i != 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, img_ch, 3, padding=1)

        self.num_res_blocks = num_res_blocks
        self.n_stages = len(ch_mults)

    def forward(self, xt, ir, vi, t, tokens):
        B, _, H, W = xt.shape
        t_map = self.time_emb(t, (H, W))
        u = torch.cat([xt, ir, vi, t_map], dim=1)
        h = self.stem(u)

        skips: List[torch.Tensor] = [h]
        idx = 0
        for i in range(self.n_stages):
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[idx](h)
                h = self.enc_dcmms[idx](h, tokens, ir, vi)
                skips.append(h)
                idx += 1
            h = self.downsamples[i](h)
            if not isinstance(self.downsamples[i], nn.Identity):
                skips.append(h)

        h = self.mid1(h)
        h = self.mid_dcmm(h, tokens, ir, vi)
        h = self.mid2(h)

        idx = 0
        for i in reversed(range(self.n_stages)):
            for _ in range(self.num_res_blocks + 1):
                s = skips.pop()
                h = torch.cat([h, s], dim=1)
                h = self.dec_blocks[idx](h)
                h = self.dec_dcmms[idx](h, tokens, ir, vi)
                idx += 1
            h = self.upsamples[self.n_stages - 1 - i](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
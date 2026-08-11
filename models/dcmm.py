
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.q_proj = nn.Linear(q_dim, out_dim)
        self.k_proj = nn.Linear(kv_dim, out_dim)
        self.v_proj = nn.Linear(kv_dim, out_dim)
        self.o_proj = nn.Linear(out_dim, out_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, kv):
        B, Nq, _ = q.shape
        Nkv = kv.shape[1]
        Q = self.q_proj(q).view(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.o_proj(out)


class DCMM(nn.Module):
    def __init__(
        self,
        feat_ch: int,
        token_dim: int,
        ir_ch: int = 1,
        vi_ch: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.feat_ch = feat_ch

        self.qg_proj = nn.Linear(feat_ch, token_dim)
        self.qc_conv = nn.Conv2d(feat_ch, token_dim, kernel_size=1)

        self.mha_g = MultiHeadAttention(token_dim, token_dim, token_dim, heads=heads)
        self.mha_c = MultiHeadAttention(token_dim, token_dim, token_dim, heads=heads)

        self.yg_to_feat = nn.Conv2d(token_dim, feat_ch, kernel_size=1)
        self.yc_to_affine = nn.Conv2d(token_dim, 2 * feat_ch, kernel_size=1)

        mod_ch = max(feat_ch, 32)
        self.mod_conv = nn.Sequential(
            nn.Conv2d(ir_ch + vi_ch, mod_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mod_ch, mod_ch, 3, padding=1),
        )
        self.alpha_conv = nn.Conv2d(mod_ch, feat_ch, kernel_size=1)
        self.irvi_to_affine = nn.Conv2d(mod_ch, 2 * feat_ch, kernel_size=1)

        self.g1 = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))
        self.g2 = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))
        self.g3 = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))

    def forward(self, feat, tokens, ir, vi):
        B, C, H, W = feat.shape
        ir_ = F.adaptive_avg_pool2d(ir, (H, W))
        vi_ = F.adaptive_avg_pool2d(vi, (H, W))

        # Global attention: Qg
        gap = feat.mean(dim=(2, 3))
        qg = self.qg_proj(gap).unsqueeze(1)
        yg = self.mha_g(qg, tokens)
        yg_map = yg.transpose(1, 2).unsqueeze(-1).expand(-1, -1, H, W)
        yg_feat = self.yg_to_feat(yg_map)


        qc_map = self.qc_conv(feat)
        qc = qc_map.flatten(2).transpose(1, 2)
        yc = self.mha_c(qc, tokens)
        yc = yc.transpose(1, 2).view(B, -1, H, W)
        text_affine = self.yc_to_affine(yc)
        gamma_text, beta_text = text_affine.chunk(2, dim=1)


        z = self.mod_conv(torch.cat([ir_, vi_], dim=1))
        alpha = torch.sigmoid(self.alpha_conv(z))
        irvi_affine = self.irvi_to_affine(z)
        gamma_irvi, beta_irvi = irvi_affine.chunk(2, dim=1)


        text_branch = feat * gamma_text + beta_text
        irvi_branch = alpha * (feat * gamma_irvi + beta_irvi)
        return feat + self.g1 * yg_feat + self.g2 * text_branch + self.g3 * irvi_branch
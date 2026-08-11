
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionConcat(nn.Module):
    def __init__(
        self,
        feat_ch: int,
        token_dim: int,
        ir_ch: int = 1,
        vi_ch: int = 3,
        text_ch: int = 64,
    ):
        super().__init__()

        if feat_ch <= 0:
            raise ValueError("feat_ch must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")

        self.feat_ch = feat_ch
        self.text_ch = min(text_ch, feat_ch)

        self.text_proj = nn.Sequential(
            nn.Linear(token_dim, self.text_ch),
            nn.SiLU(),
        )

        input_ch = feat_ch + self.text_ch + ir_ch + vi_ch
        hidden_ch = max(feat_ch // 2, 32)

        self.fuse = nn.Sequential(
            nn.Conv2d(input_ch, hidden_ch, kernel_size=1),
            nn.GroupNorm(min(8, hidden_ch), hidden_ch),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, feat_ch, kernel_size=3, padding=1),
        )


        self.gate = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))

    def forward(
        self,
        feat: torch.Tensor,
        tokens: torch.Tensor,
        ir: torch.Tensor,
        vi: torch.Tensor,
    ) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError("feat must be [B, C, H, W]")
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B, token_len, token_dim]")
        if ir.ndim != 4 or vi.ndim != 4:
            raise ValueError("ir and vi must be [B, C, H, W]")

        batch, _, height, width = feat.shape

        if tokens.shape[0] != batch:
            raise ValueError(
                "feat and tokens must have the same batch size"
            )

        text_vec = tokens.mean(dim=1)
        text_vec = self.text_proj(text_vec)
        text_map = text_vec[:, :, None, None].expand(
            -1, -1, height, width
        )

        ir_map = F.adaptive_avg_pool2d(ir, (height, width))
        vi_map = F.adaptive_avg_pool2d(vi, (height, width))

        merged = torch.cat(
            [feat, text_map, ir_map, vi_map],
            dim=1,
        )

        update = self.fuse(merged)
        return feat + self.gate * update

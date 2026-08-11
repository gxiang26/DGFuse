
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_concat import ConditionConcat


class TimeEmbeddingDCMM(nn.Module):
    def __init__(
        self,
        sinu_dim: int = 128,
        out_ch: int = 4,
    ):
        super().__init__()
        self.sinu_dim = sinu_dim
        self.out_ch = out_ch

        self.mlp = nn.Sequential(
            nn.Linear(sinu_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_ch),
        )

    def forward(
        self,
        t: torch.Tensor,
        spatial: Tuple[int, int],
    ) -> torch.Tensor:
        device = t.device
        half = self.sinu_dim // 2

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device).float()
            / half
        )

        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat(
            [args.sin(), args.cos()],
            dim=-1,
        )

        emb = self.mlp(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)

        return emb.expand(
            -1,
            -1,
            spatial[0],
            spatial[1],
        )


class ResBlockDCMM(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
    ):
        super().__init__()

        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            3,
            padding=1,
        )

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(
            F.silu(self.norm1(x))
        )
        h = self.conv2(
            F.silu(self.norm2(h))
        )
        return h + self.skip(x)


class ConditionalUNetDCMM(nn.Module):
    """
    DCMM-ablation U-Net.

    The name is different from the original ``ConditionalUNet``, so this
    file can coexist with the full model without overwriting it.
    """

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


        del attn_heads

        self.img_ch = img_ch
        self.ir_ch = ir_ch
        self.vi_ch = vi_ch
        self.time_ch = time_ch

        input_ch = img_ch + ir_ch + vi_ch + time_ch

        self.time_emb = TimeEmbeddingDCMM(
            sinu_dim=128,
            out_ch=time_ch,
        )

        self.stem = nn.Conv2d(
            input_ch,
            base_ch,
            3,
            padding=1,
        )


        self.enc_blocks = nn.ModuleList()
        self.enc_conditions = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_ch
        encoder_channels = [ch]

        for stage_idx, multiplier in enumerate(ch_mults):
            out_ch = base_ch * multiplier

            for _ in range(num_res_blocks):
                self.enc_blocks.append(
                    ResBlockDCMM(ch, out_ch)
                )

                self.enc_conditions.append(
                    ConditionConcat(
                        feat_ch=out_ch,
                        token_dim=token_dim,
                        ir_ch=ir_ch,
                        vi_ch=vi_ch,
                    )
                )

                ch = out_ch
                encoder_channels.append(ch)

            if stage_idx != len(ch_mults) - 1:
                self.downsamples.append(
                    nn.Conv2d(
                        ch,
                        ch,
                        3,
                        stride=2,
                        padding=1,
                    )
                )
                encoder_channels.append(ch)
            else:
                self.downsamples.append(
                    nn.Identity()
                )


        self.mid1 = ResBlockDCMM(ch, ch)

        self.mid_condition = ConditionConcat(
            feat_ch=ch,
            token_dim=token_dim,
            ir_ch=ir_ch,
            vi_ch=vi_ch,
        )

        self.mid2 = ResBlockDCMM(ch, ch)


        self.dec_blocks = nn.ModuleList()
        self.dec_conditions = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for stage_idx, multiplier in reversed(
            list(enumerate(ch_mults))
        ):
            out_ch = base_ch * multiplier

            for _ in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()

                self.dec_blocks.append(
                    ResBlockDCMM(
                        ch + skip_ch,
                        out_ch,
                    )
                )

                self.dec_conditions.append(
                    ConditionConcat(
                        feat_ch=out_ch,
                        token_dim=token_dim,
                        ir_ch=ir_ch,
                        vi_ch=vi_ch,
                    )
                )

                ch = out_ch

            if stage_idx != 0:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        ch,
                        ch,
                        4,
                        stride=2,
                        padding=1,
                    )
                )
            else:
                self.upsamples.append(
                    nn.Identity()
                )

        self.out_norm = nn.GroupNorm(
            min(8, ch),
            ch,
        )
        self.out_conv = nn.Conv2d(
            ch,
            img_ch,
            3,
            padding=1,
        )

        self.num_res_blocks = num_res_blocks
        self.n_stages = len(ch_mults)

    def forward(
        self,
        xt: torch.Tensor,
        ir: torch.Tensor,
        vi: torch.Tensor,
        t: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = xt.shape

        t_map = self.time_emb(
            t,
            (height, width),
        )

        u = torch.cat(
            [xt, ir, vi, t_map],
            dim=1,
        )
        h = self.stem(u)

        skips: List[torch.Tensor] = [h]
        block_idx = 0

        for stage_idx in range(self.n_stages):
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[block_idx](h)

                h = self.enc_conditions[block_idx](
                    h,
                    tokens,
                    ir,
                    vi,
                )

                skips.append(h)
                block_idx += 1

            h = self.downsamples[stage_idx](h)

            if not isinstance(
                self.downsamples[stage_idx],
                nn.Identity,
            ):
                skips.append(h)

        h = self.mid1(h)

        h = self.mid_condition(
            h,
            tokens,
            ir,
            vi,
        )

        h = self.mid2(h)

        block_idx = 0

        for stage_idx in reversed(
            range(self.n_stages)
        ):
            for _ in range(
                self.num_res_blocks + 1
            ):
                skip = skips.pop()

                h = torch.cat(
                    [h, skip],
                    dim=1,
                )

                h = self.dec_blocks[block_idx](h)

                h = self.dec_conditions[block_idx](
                    h,
                    tokens,
                    ir,
                    vi,
                )

                block_idx += 1

            upsample_idx = (
                self.n_stages
                - 1
                - stage_idx
            )

            h = self.upsamples[
                upsample_idx
            ](h)

        h = F.silu(
            self.out_norm(h)
        )

        return self.out_conv(h)

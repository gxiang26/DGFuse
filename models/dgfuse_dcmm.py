
from typing import List

import torch
import torch.nn as nn

from .text_encoder import DemandTextEncoder
from .unet_dcmm import ConditionalUNetDCMM


class DGFuseDCMM(nn.Module):


    def __init__(
        self,
        cfg,
        device: str = "cuda:1",
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device

        self.text_encoder = DemandTextEncoder(
            clip_model_name=cfg.clip_model_name,
            token_len=cfg.token_len,
            token_dim=cfg.token_dim,
            device=device,
        )

        self.unet = ConditionalUNetDCMM(
            img_ch=cfg.img_channels,
            ir_ch=cfg.ir_channels,
            vi_ch=cfg.vi_channels,
            time_ch=cfg.time_channels,
            base_ch=cfg.base_ch,
            ch_mults=cfg.ch_mults,
            num_res_blocks=cfg.num_res_blocks,
            token_dim=cfg.token_dim,
            attn_heads=cfg.attn_heads,
        )

        self.sigma_min = cfg.sigma_min

    def training_step(
        self,
        gt: torch.Tensor,
        ir: torch.Tensor,
        vi: torch.Tensor,
        texts: List[str],
    ):
        batch = gt.shape[0]
        device = gt.device

        t = torch.rand(
            batch,
            device=device,
        )

        eps = torch.randn_like(gt)

        t_map = t.view(
            batch,
            1,
            1,
            1,
        )

        xt = (
            1
            - (1 - self.sigma_min) * t_map
        ) * eps + t_map * gt

        v_star = (
            gt
            - (1 - self.sigma_min) * eps
        )

        tokens = self.text_encoder(texts)

        v_pred = self.unet(
            xt,
            ir,
            vi,
            t,
            tokens,
        )

        x_hat = (
            v_pred
            + (1 - self.sigma_min) * eps
        )

        return v_pred, v_star, x_hat

    @torch.no_grad()
    def sample(
        self,
        ir: torch.Tensor,
        vi: torch.Tensor,
        texts: List[str],
        steps: int = 20,
        integrator: str = "rk4",
    ) -> torch.Tensor:
        batch = ir.shape[0]
        device = ir.device
        height, width = ir.shape[-2:]

        x = torch.randn(
            batch,
            self.cfg.img_channels,
            height,
            width,
            device=device,
        )

        tokens = self.text_encoder(texts)
        dt = 1.0 / steps

        def velocity(
            x_in: torch.Tensor,
            t_scalar: float,
        ) -> torch.Tensor:
            t_vec = torch.full(
                (batch,),
                t_scalar,
                device=device,
            )

            return self.unet(
                x_in,
                ir,
                vi,
                t_vec,
                tokens,
            )

        for step_idx in range(steps):
            t = step_idx * dt

            if integrator == "euler":
                x = x + dt * velocity(x, t)

            elif integrator == "rk4":
                k1 = velocity(x, t)

                k2 = velocity(
                    x + 0.5 * dt * k1,
                    t + 0.5 * dt,
                )

                k3 = velocity(
                    x + 0.5 * dt * k2,
                    t + 0.5 * dt,
                )

                k4 = velocity(
                    x + dt * k3,
                    t + dt,
                )

                x = x + (dt / 6.0) * (
                    k1
                    + 2 * k2
                    + 2 * k3
                    + k4
                )

            else:
                raise ValueError(
                    f"Unknown integrator {integrator!r}"
                )

        return x

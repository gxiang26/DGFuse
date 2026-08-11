
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # --------------------------- Data ---------------------------
    llvip_vi_dir: str = r""
    llvip_ir_dir: str = r""
    llvip_metric_root: str = ""
    llvip_fusion_root: str = ""
    llvip_ext_vi: str = "" #jpg
    llvip_ext_ir: str = ""

    fmb_vi_dir: str = r""
    fmb_ir_dir: str = r""
    fmb_metric_root: str = ""
    fmb_fusion_root: str = ""
    fmb_ext_vi: str = ""#png
    fmb_ext_ir: str = ""

    crop_size: int = 256
    num_workers: int = 0

    img_channels: int = 3
    ir_channels: int = 1
    vi_channels: int = 3
    time_channels: int = 4

    base_ch: int = 64
    ch_mults: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_heads: int = 4

    token_len: int = 16
    token_dim: int = 256
    clip_model_name: str = "ViT-B/32"
    sigma_min: float = 1e-3


    batch_size: int = 1
    epochs: int = 850
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    lambda_fm: float = 1.3
    lambda_grad: float = 0.5

    uncond_prob: float = 0.1


    steps_inference: int = 4
    integrator: str = "euler"

    # --------------------------- IO ---------------------------
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    log_every: int = 50
    save_every_epoch: int = 10
    seed: int = 42


CFG = Config()
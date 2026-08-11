
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from config import CFG
from models import DGFuse


def pil_to_tensor(img: Image.Image, channels: int) -> torch.Tensor:
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.clamp(-1, 1) + 1) / 2
    t = (t * 255).round().byte().cpu().numpy()
    if t.shape[0] == 1:
        return Image.fromarray(t[0], mode="L")
    return Image.fromarray(t.transpose(1, 2, 0), mode="RGB")


def pad_to_multiple(x: torch.Tensor, m: int = 16):
    _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    top, bot = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    x_p = torch.nn.functional.pad(
        x.unsqueeze(0), (left, right, top, bot), mode="reflect"
    ).squeeze(0)
    return x_p, (top, bot, left, right)


def unpad(x: torch.Tensor, pads):
    top, bot, left, right = pads
    return x[..., top:x.shape[-2] - bot if bot else x.shape[-2],
                  left:x.shape[-1] - right if right else x.shape[-1]]


def load_model(ckpt_path: str, device: str, use_ema: bool = True) -> DGFuse:
    model = DGFuse(CFG, device=device).to(device)
    state = torch.load(ckpt_path, map_location=device)

    if use_ema and "ema" in state and state["ema"]:
        merged = dict(state["model"])
        merged.update(state["ema"])
        model.load_state_dict(merged)
        print("[test] loaded EMA weights")
    else:
        model.load_state_dict(state["model"] if "model" in state else state)
        print("[test] loaded raw weights")
    model.eval()
    return model


@torch.no_grad()
def fuse_pair(model, vi_path, ir_path, text, steps, integrator, device):
    vi_pil = Image.open(vi_path)
    ir_pil = Image.open(ir_path)
    W, H = vi_pil.size
    if ir_pil.size != (W, H):
        ir_pil = ir_pil.resize((W, H), Image.BILINEAR)

    vi = pil_to_tensor(vi_pil, 3).to(device) * 2 - 1
    ir = pil_to_tensor(ir_pil, 1).to(device) * 2 - 1

    vi_p, pads = pad_to_multiple(vi, m=16)
    ir_p, _ = pad_to_multiple(ir, m=16)

    out = model.sample(ir_p.unsqueeze(0), vi_p.unsqueeze(0),
                       [text], steps=steps, integrator=integrator)
    out = unpad(out.squeeze(0), pads)
    return tensor_to_pil(out)


def list_pairs(vi_dir: Path, ir_dir: Path) -> List[Tuple[Path, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    vi_map = {p.stem: p for p in vi_dir.iterdir() if p.suffix.lower() in exts}
    ir_map = {p.stem: p for p in ir_dir.iterdir() if p.suffix.lower() in exts}
    common = sorted(vi_map.keys() & ir_map.keys())
    return [(vi_map[k], ir_map[k]) for k in common]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vi", type=str, default="")
    parser.add_argument("--ir", type=str, default="")
    parser.add_argument("--vi-dir", type=str, default="")
    parser.add_argument("--ir-dir", type=str, default="")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--out", type=str, default="./results")
    parser.add_argument("--steps", type=int, default=CFG.steps_inference)
    parser.add_argument("--integrator", type=str, default=CFG.integrator,
                        choices=["euler", "rk4"])
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device, use_ema=not args.no_ema)

    if args.vi and args.ir:
        pairs = [(Path(args.vi), Path(args.ir))]
    elif args.vi_dir and args.ir_dir:
        pairs = list_pairs(Path(args.vi_dir), Path(args.ir_dir))
        print(f"[test] found {len(pairs)} pairs")
    else:
        raise SystemExit("Provide either --vi/--ir or --vi-dir/--ir-dir")

    for vi_p, ir_p in pairs:
        img = fuse_pair(model, vi_p, ir_p, args.text, args.steps,
                        args.integrator, device)
        save_path = out_dir / f"{vi_p.stem}.png"
        img.save(save_path)
        print(f"[test] wrote {save_path}")


if __name__ == "__main__":
    main()
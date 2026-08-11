import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from models import DGFuse
from losses import total_loss
from datasets import build_train_datasets, collate
from demand import DemandRouter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow


def save_ckpt(path, model, ema, optimizer, epoch, step):
    torch.save({
        "model": model.state_dict(),
        "ema":   ema.state_dict(),
        "opt":   optimizer.state_dict(),
        "epoch": epoch,
        "step":  step,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    model = DGFuse(cfg, device=device).to(device)

    demand_router = DemandRouter(
        clip_model=model.text_encoder.clip_model,
        tau=0.07, device=device,
    )

    train_set = build_train_datasets(cfg, demand_router)
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    print(f"[train] {len(train_set)} samples · {len(loader)} steps/epoch")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    ema = EMA(model, decay=0.999)

    start_epoch, global_step = 0, 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        optimizer.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["step"]
        print(f"[train] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        for batch in loader:
            vi = batch["vi"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            texts = batch["text"]

            v_pred, v_star, x_hat = model.training_step(gt, ir, vi, texts)
            loss, parts = total_loss(
                v_pred, v_star, x_hat, ir, vi,
                lambda_fm=cfg.lambda_fm, lambda_grad=cfg.lambda_grad,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            if global_step % cfg.log_every == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/fm",    parts["l_fm"].item(),   global_step)
                writer.add_scalar("loss/grad",  parts["l_grad"].item(), global_step)
                print(f"ep{epoch:04d} step{global_step:07d} "
                      f"loss={loss.item():.4f} fm={parts['l_fm'].item():.4f} "
                      f"grad={parts['l_grad'].item():.4f}")
            global_step += 1

        print(f"[epoch {epoch}] took {time.time()-t0:.1f}s")

        if (epoch + 1) % cfg.save_every_epoch == 0 or epoch + 1 == cfg.epochs:
            path = Path(cfg.ckpt_dir) / f"dgfuse_ep{epoch+1:04d}.pt"
            save_ckpt(path, model, ema, optimizer, epoch, global_step)
            print(f"[train] saved {path}")

    save_ckpt(Path(cfg.ckpt_dir) / "dgfuse_final.pt",
              model, ema, optimizer, cfg.epochs - 1, global_step)
    writer.close()


if __name__ == "__main__":
    main()
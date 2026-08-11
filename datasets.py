

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from demand import FusionSelector, METRIC_PROTOTYPES


DATASET_ALLOWED_METRICS = {
    "LLVIP": ["SSIM", "QABF", "VIF", "SD", "AP"],
    "FMB":   ["SSIM", "QABF", "VIF", "SD", "mIoU"],
}


def sample_demand_for(dataset: str, uncond_prob: float) -> Optional[str]:
    if random.random() < uncond_prob:
        return None
    metric = random.choice(DATASET_ALLOWED_METRICS[dataset])
    return METRIC_PROTOTYPES[metric][0]


class PairedRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, vi, ir, gt):
        W, H = vi.size
        if H < self.size or W < self.size:
            vi = vi.resize((max(W, self.size), max(H, self.size)))
            ir = ir.resize((max(W, self.size), max(H, self.size)))
            gt = gt.resize((max(W, self.size), max(H, self.size)))
            W, H = vi.size
        x = random.randint(0, W - self.size)
        y = random.randint(0, H - self.size)
        box = (x, y, x + self.size, y + self.size)
        return vi.crop(box), ir.crop(box), gt.crop(box)


def pil_to_tensor(img: Image.Image, channels: int) -> torch.Tensor:
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class IVIFDataset(Dataset):
    def __init__(
        self,
        selector: FusionSelector,
        vi_dir: str,
        ir_dir: str,
        image_ids: List[str],
        vi_ext: str = "png",
        ir_ext: str = "png",
        crop_size: int = 256,
        uncond_prob: float = 0.1,
        augment: bool = True,
    ):
        self.sel = selector
        self.vi_dir = Path(vi_dir)
        self.ir_dir = Path(ir_dir)
        self.ids = image_ids
        self.vi_ext = vi_ext.lstrip(".")
        self.ir_ext = ir_ext.lstrip(".")
        self.crop_size = crop_size
        self.uncond_prob = uncond_prob
        self.augment = augment
        self.crop = PairedRandomCrop(crop_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        text = sample_demand_for(self.sel.dataset, self.uncond_prob)

        info = self.sel.select(iid, demand_text=text)
        vi = Image.open(self.vi_dir / f"{iid}.{self.vi_ext}").convert("RGB")
        ir = Image.open(self.ir_dir / f"{iid}.{self.ir_ext}").convert("L")
        gt = Image.open(info["path"]).convert("RGB")

        if self.augment:
            vi, ir, gt = self.crop(vi, ir, gt)
            if random.random() < 0.5:
                vi, ir, gt = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in (vi, ir, gt)]

        vi_t = pil_to_tensor(vi, 3) * 2 - 1
        ir_t = pil_to_tensor(ir, 1) * 2 - 1
        gt_t = pil_to_tensor(gt, 3) * 2 - 1

        return {
            "vi": vi_t, "ir": ir_t, "gt": gt_t,
            "text": text if text is not None else "",
            "image_id": iid, "dataset": self.sel.dataset,
            "gt_method": info["method"],
            "metric_used": info.get("metric_used") or "",
        }


def collate(batch):
    out = {}
    out["vi"] = torch.stack([b["vi"] for b in batch])
    out["ir"] = torch.stack([b["ir"] for b in batch])
    out["gt"] = torch.stack([b["gt"] for b in batch])
    out["text"] = [b["text"] for b in batch]
    out["image_id"] = [b["image_id"] for b in batch]
    out["dataset"] = [b["dataset"] for b in batch]
    out["gt_method"] = [b["gt_method"] for b in batch]
    out["metric_used"] = [b["metric_used"] for b in batch]
    return out


def discover_ids(vi_dir: str, vi_ext: str = "png") -> List[str]:
    return sorted(p.stem for p in Path(vi_dir).glob(f"*.{vi_ext}"))


def build_train_datasets(cfg, router) -> ConcatDataset:
    llvip_sel = FusionSelector(
        metric_root=cfg.llvip_metric_root,
        fusion_root=cfg.llvip_fusion_root,
        router=router, dataset="LLVIP", id_col="image",
    )
    fmb_sel = FusionSelector(
        metric_root=cfg.fmb_metric_root,
        fusion_root=cfg.fmb_fusion_root,
        router=router, dataset="FMB", id_col="image",
    )

    llvip_ids = discover_ids(cfg.llvip_vi_dir, cfg.llvip_ext_vi)
    fmb_ids   = discover_ids(cfg.fmb_vi_dir,   cfg.fmb_ext_vi)
    llvip_ids = [i for i in llvip_ids if i in llvip_sel._cache]
    fmb_ids   = [i for i in fmb_ids   if i in fmb_sel._cache]

    llvip_ds = IVIFDataset(llvip_sel, cfg.llvip_vi_dir, cfg.llvip_ir_dir,
                           llvip_ids, vi_ext=cfg.llvip_ext_vi,
                           ir_ext=cfg.llvip_ext_ir,
                           crop_size=cfg.crop_size, uncond_prob=cfg.uncond_prob)
    fmb_ds = IVIFDataset(fmb_sel, cfg.fmb_vi_dir, cfg.fmb_ir_dir,
                         fmb_ids, vi_ext=cfg.fmb_ext_vi,
                         ir_ext=cfg.fmb_ext_ir,
                         crop_size=cfg.crop_size, uncond_prob=cfg.uncond_prob)
    return ConcatDataset([llvip_ds, fmb_ds])
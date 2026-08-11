

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import clip


METRIC_KEYS: List[str] = ["SSIM", "QABF", "VIF", "SD", "AP", "mIoU"]

METRIC_PROTOTYPES: Dict[str, List[str]] = {
    "SSIM": ["Generate fused images with more texture details"],
    "QABF": ["Generate fused images with more prominent infrared targets"],
    "VIF":  ["Generate fused images with better visual perception"],
    "SD":   ["Generate fused images with higher contrast"],
    "AP":   ["Generate fused images with better detection performance"],
    "mIoU": ["Generate fused images with better segmentation performance"],
}

METRIC_TO_DATASET: Dict[str, Optional[str]] = {
    "SSIM": None,
    "QABF": None,
    "VIF":  None,
    "SD":   None,
    "AP":   "LLVIP",
    "mIoU": "FMB",
}


class DemandRouter:
    def __init__(
        self,
        clip_model,
        tau: float = 0.07,
        device: str = "cuda",
        prototypes: Dict[str, List[str]] = None,
        metric_keys: List[str] = None,
    ):
        self.clip_model = clip_model
        self.tau = tau
        self.device = device
        self.prototypes = prototypes or METRIC_PROTOTYPES
        self.metric_keys = metric_keys or METRIC_KEYS

        self._clip_zero_warned = False
        self.anchors = self._encode_prototypes()

    def _safe_normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        n = x.norm(dim=dim, keepdim=True)
        return x / n.clamp_min(1e-8)

    def _one_hot(self, metric: str) -> torch.Tensor:
        w = torch.zeros(len(self.metric_keys), device=self.device)
        idx = self.metric_keys.index(metric)
        w[idx] = 1.0
        return w

    def _fallback_metric(self, demand_text: str) -> str:
        s = (demand_text or "").lower()


        for metric, sents in self.prototypes.items():
            for p in sents:
                if s.strip() == p.lower().strip():
                    return metric


        if "texture" in s or "detail" in s or "structure" in s:
            return "SSIM"
        if "infrared" in s or "target" in s or "salient" in s:
            return "QABF"
        if "visual" in s or "perception" in s or "natural" in s:
            return "VIF"
        if "contrast" in s or "brightness" in s:
            return "SD"
        if "detection" in s or "detect" in s or "ap" in s:
            return "AP"
        if "segmentation" in s or "segment" in s or "miou" in s:
            return "mIoU"



        return "SSIM"

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        emb = self.clip_model.encode_text(tokens).float()

        if emb.detach().abs().max().item() == 0.0:
            if not self._clip_zero_warned:
                print(
                    "[WARN/DemandRouter] CLIP encode_text returned all-zero. "
                    "Router will use keyword/prototype fallback."
                )
                self._clip_zero_warned = True

        return emb

    @torch.no_grad()
    def _encode_prototypes(self) -> torch.Tensor:
        anchors = []

        for k in self.metric_keys:
            sents = self.prototypes[k]
            emb = self._encode_text(sents)

            if emb.detach().abs().max().item() == 0.0:

                anchor = emb.mean(dim=0, keepdim=True)
            else:
                emb = self._safe_normalize(emb, dim=-1)
                anchor = self._safe_normalize(emb.mean(dim=0, keepdim=True), dim=-1)

            anchors.append(anchor)

        anchors = torch.cat(anchors, dim=0)

        if torch.isnan(anchors).any() or torch.isinf(anchors).any():
            anchors = torch.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0)

        return anchors

    @torch.no_grad()
    def get_weights(self, demand_text: str) -> torch.Tensor:
        emb = self._encode_text([demand_text])

        if emb.detach().abs().max().item() == 0.0:
            metric = self._fallback_metric(demand_text)
            return self._one_hot(metric)

        emb = self._safe_normalize(emb, dim=-1)

        if self.anchors.detach().abs().max().item() == 0.0:
            metric = self._fallback_metric(demand_text)
            return self._one_hot(metric)

        sims = (emb @ self.anchors.T).squeeze(0)

        if torch.isnan(sims).any() or torch.isinf(sims).any():
            metric = self._fallback_metric(demand_text)
            return self._one_hot(metric)

        return F.softmax(sims / self.tau, dim=-1)

    def get_weights_dict(self, demand_text: str) -> Dict[str, float]:
        w = self.get_weights(demand_text).detach().cpu().tolist()
        return {k: float(v) for k, v in zip(self.metric_keys, w)}

    def get_top_metric(self, demand_text: str) -> str:
        w = self.get_weights_dict(demand_text)
        return max(w, key=w.get)

    def route_dataset(self, demand_text: str) -> Optional[str]:
        return METRIC_TO_DATASET.get(self.get_top_metric(demand_text))
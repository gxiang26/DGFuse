
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import numpy as np
import pandas as pd


DATASET_METRICS = {
    "LLVIP": ["SSIM", "QABF", "VIF", "SD", "AP"],
    "FMB":   ["SSIM", "QABF", "VIF", "SD", "mIoU"],
}

CSV_ALIASES = {
    "SSIM": ["SSIM", "SSIM_VI", "ssim_vi"],
    "QABF": ["QABF", "Qabf_IR", "qabf_ir"],
    "VIF":  ["VIF",  "VIF_VI",  "vif_vi"],
    "SD":   ["SD"],
    "AP":   ["AP", "mAP", "AP50_95"],
    "mIoU": ["mIoU", "miou", "miou_std"],
}


def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


class FusionSelector:
    def __init__(
        self,
        metric_root,
        fusion_root,
        router,
        dataset,
        id_col="image",
        methods=None,
        metric_cols=None,
        metric_alias=None,
        method_exts=None,
        path_resolver=None,
        strict_dataset_routing=True,
        verbose=True,
    ):
        self.router = router
        self.dataset = dataset
        self.metric_root = Path(metric_root)
        self.fusion_root = Path(fusion_root)
        self.id_col = id_col
        self.metric_cols = metric_cols or DATASET_METRICS[dataset]
        self.metric_alias = metric_alias or {}
        self.path_resolver = path_resolver
        self.strict_dataset_routing = strict_dataset_routing

        if methods is None:
            methods = sorted(p.stem for p in self.metric_root.glob("*.csv"))
        if not methods:
            raise FileNotFoundError(f"{self.metric_root} 下未找到任何 *.csv")
        self.methods = methods


        self.method_exts = {}
        for m in self.methods:
            if method_exts and m in method_exts:
                self.method_exts[m] = method_exts[m].lstrip(".")
                continue
            mdir = self.fusion_root / m
            ext = None
            if mdir.is_dir():
                for p in mdir.iterdir():
                    if p.is_file() and p.suffix:
                        ext = p.suffix.lstrip(".")
                        break
            self.method_exts[m] = ext or "png"

        self.df_raw = self._load_and_merge(methods)
        self.df = self._normalize_per_image(self.df_raw)
        self._cache = {
            str(iid): g.reset_index(drop=True)
            for iid, g in self.df.groupby("image_id")
        }

        if verbose:
            print(f"[FusionSelector/{dataset}] "
                  f"{len(self._cache)} images · "
                  f"{len(self.methods)} methods · "
                  f"metrics={self.metric_cols}")
            print(f"  method_exts: {self.method_exts}")


    def _load_and_merge(self, methods):
        frames = []
        for m in methods:
            csv_path = self.metric_root / f"{m}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            df = pd.read_csv(csv_path)
            if self.id_col not in df.columns:
                raise ValueError(
                    f"{csv_path} 缺少 id 列 '{self.id_col}'. 实际列: {list(df.columns)}"
                )
            image_ids = df[self.id_col].astype(str).map(lambda s: Path(s).stem)
            out = pd.DataFrame({"image_id": image_ids.values, "method": m})
            for std in self.metric_cols:
                aliases = CSV_ALIASES.get(std, [std])
                col = _find_col(df, aliases)
                out[std] = df[col].astype(float).values if col else np.nan
            frames.append(out)
        return pd.concat(frames, ignore_index=True)

    def _normalize_per_image(self, df):

        parts = []
        for _, g in df.groupby("image_id"):
            g = g.copy()
            for m in self.metric_cols:
                v = g[m].astype(float).values
                ok = ~np.isnan(v)
                if ok.sum() < 2:
                    n = np.where(ok, 0.5, np.nan)
                else:
                    vmin, vmax = v[ok].min(), v[ok].max()
                    if vmax - vmin < 1e-8:
                        n = np.full_like(v, 0.5, dtype=float)
                        n[~ok] = np.nan
                    else:
                        n = (v - vmin) / (vmax - vmin)
                        n[~ok] = np.nan
                g[m + "_n"] = n
            parts.append(g)
        return pd.concat(parts, ignore_index=True)


    def _resolve_path(self, method, image_id):
        if self.path_resolver is not None:
            return Path(self.path_resolver(method, image_id))
        ext = self.method_exts.get(method, "png")
        return self.fusion_root / method / f"{image_id}.{ext}"

    def _norm_id(self, image_id):
        return Path(str(image_id)).stem


    def _pick_top_metric(self, demand_text: str) -> str:

        if self.router is None:
            raise RuntimeError("传入 demand_text 但 router=None")
        w_raw = self.router.get_weights_dict(demand_text)
        if self.metric_alias:
            w_raw = {self.metric_alias.get(k, k): v for k, v in w_raw.items()}
        top = max(w_raw, key=w_raw.get)

        if top not in self.metric_cols:
            if self.strict_dataset_routing:
                raise ValueError(
                    f"Demand '{demand_text}' 路由到指标 '{top}',"
                    f"但 '{self.dataset}' 数据集不包含该指标 "
                    f"(可用: {self.metric_cols})。应使用对应的数据集训练。"
                )

            avail = {m: w_raw[m] for m in self.metric_cols if m in w_raw}
            top = max(avail, key=avail.get) if avail else self.metric_cols[0]
        return top

    def _uniform_weights(self):
        n = len(self.metric_cols)
        return {m: 1.0 / n for m in self.metric_cols}


    def select(self, image_id, demand_text=None, return_ranking=False):
        key = self._norm_id(image_id)
        if key not in self._cache:
            raise KeyError(f"image_id '{image_id}' (stem='{key}') 不在任何方法的 CSV 中")
        sub = self._cache[key]
        FORCE_METHOD = None

        if FORCE_METHOD is not None:
            matched = sub[sub["method"] == FORCE_METHOD]
            if len(matched) > 0:
                return {
                    "image_id": key,
                    "dataset": self.dataset,
                    "method": FORCE_METHOD,
                    "path": str(self._resolve_path(FORCE_METHOD, key)),
                    "score": 1.0,
                    "weights": {},
                    "mode": "forced",
                    "metric_used": None,
                }

        if demand_text is not None and demand_text.strip():

            top = self._pick_top_metric(demand_text)
            raw = sub[top].astype(float).values
            scores = np.where(np.isnan(raw), -np.inf, raw)
            mode = "text"
            metric_used = top
            weights_info = {top: 1.0}
        else:

            weights = self._uniform_weights()
            scores = np.zeros(len(sub), dtype=float)
            for m, w in weights.items():
                v = np.nan_to_num(sub[m + "_n"].astype(float).values, nan=0.0)
                scores += w * v
            mode = "uniform"
            metric_used = None
            weights_info = weights

        best = int(np.argmax(scores))
        best_method = str(sub.iloc[best]["method"])

        out = {
            "image_id": key,
            "dataset": self.dataset,
            "method": best_method,
            "path": str(self._resolve_path(best_method, key)),
            "score": float(scores[best]) if np.isfinite(scores[best]) else float("nan"),
            "metric_used": metric_used,
            "weights": weights_info,
            "mode": mode,
        }
        if return_ranking:
            order = np.argsort(-scores)
            out["ranking"] = [
                (str(sub.iloc[i]["method"]),
                 float(scores[i]) if np.isfinite(scores[i]) else float("nan"))
                for i in order
            ]
        return out

    def select_batch(self, image_ids, demand_text=None):

        use_text = demand_text is not None and demand_text.strip()
        if use_text:
            top = self._pick_top_metric(demand_text)
            mode = "text"
        else:
            weights = self._uniform_weights()
            mode = "uniform"

        results = []
        for iid in image_ids:
            key = self._norm_id(iid)
            sub = self._cache[key]
            if use_text:
                raw = sub[top].astype(float).values
                scores = np.where(np.isnan(raw), -np.inf, raw)
                metric_used = top
                weights_info = {top: 1.0}
            else:
                scores = np.zeros(len(sub), dtype=float)
                for m, w in weights.items():
                    v = np.nan_to_num(sub[m + "_n"].astype(float).values, nan=0.0)
                    scores += w * v
                metric_used = None
                weights_info = weights

            best = int(np.argmax(scores))
            best_method = str(sub.iloc[best]["method"])
            results.append({
                "image_id": key,
                "dataset": self.dataset,
                "method": best_method,
                "path": str(self._resolve_path(best_method, key)),
                "score": float(scores[best]) if np.isfinite(scores[best]) else float("nan"),
                "metric_used": metric_used,
                "weights": weights_info,
                "mode": mode,
            })
        return results
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _ModelSpec:
    patch_size: int
    feature_dim: int


_DINOV2_SPECS = {
    "dinov2_vits14": _ModelSpec(patch_size=14, feature_dim=384),
    "dinov2_vitb14": _ModelSpec(patch_size=14, feature_dim=768),
    "dinov2_vitl14": _ModelSpec(patch_size=14, feature_dim=1024),
    "dinov2_vitg14": _ModelSpec(patch_size=14, feature_dim=1536),
}

_DINOV3_SPECS = {
    "dinov3_vitl16": _ModelSpec(patch_size=16, feature_dim=1024),
}


@dataclass(frozen=True)
class DinoFeatureMap:
    """DINO patch-token feature grid for a batch of camera images."""

    features: torch.Tensor
    patch_size: int
    image_height: int
    image_width: int
    effective_height: int
    effective_width: int
    model_name: str
    selected_layers: tuple[int, ...] | None

    @property
    def patch_height(self) -> int:
        return int(self.features.shape[1])

    @property
    def patch_width(self) -> int:
        return int(self.features.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])


class DinoFeatureExtractor:
    """Extract DINO patch-token grids from RGB images.

    The projection step is intentionally separate so the same image features can
    be reused for every object/segment in a frame.
    """

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vits14",
        device: str | torch.device | None = None,
        selected_layers: Sequence[int] | None = None,
        dinov3_repo: str | Path | None = None,
        dinov3_weights: str | Path | None = None,
        torchhub_repo: str = "facebookresearch/dinov2",
        use_half: bool = False,
    ) -> None:
        self.model_name = str(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_half = bool(use_half)
        self.torchhub_repo = str(torchhub_repo)

        if self.model_name == "dummy":
            self.spec = _ModelSpec(patch_size=16, feature_dim=8)
            self.selected_layers = None
            self.model = None
            self.output_dim = self.spec.feature_dim
            return

        if self.model_name in _DINOV3_SPECS:
            self.spec = _DINOV3_SPECS[self.model_name]
            self.selected_layers = tuple(selected_layers) if selected_layers is not None else (4, 11, 17, 23)
            self.model = self._load_dinov3(dinov3_repo=dinov3_repo, dinov3_weights=dinov3_weights)
        elif self.model_name in _DINOV2_SPECS:
            self.spec = _DINOV2_SPECS[self.model_name]
            self.selected_layers = tuple(selected_layers) if selected_layers is not None else None
            self.model = torch.hub.load(self.torchhub_repo, self.model_name, trust_repo=True)
        else:
            supported = sorted([*_DINOV2_SPECS.keys(), *_DINOV3_SPECS.keys(), "dummy"])
            raise ValueError(f"Unsupported DINO model {self.model_name!r}. Supported: {supported}")

        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        layer_count = len(self.selected_layers) if self.selected_layers is not None else 1
        self.output_dim = int(self.spec.feature_dim * layer_count)

    def _load_dinov3(
        self,
        *,
        dinov3_repo: str | Path | None,
        dinov3_weights: str | Path | None,
    ) -> torch.nn.Module:
        repo = Path(dinov3_repo or "/workspace/PointWorld/third_party/dinov3").expanduser()
        hubconf = repo / "hubconf.py"
        if not hubconf.exists():
            raise FileNotFoundError(
                f"DINOv3 hubconf.py was not found at {hubconf}. "
                "PointWorld expects a local DINOv3 checkout under third_party/dinov3."
            )

        weights = Path(dinov3_weights).expanduser() if dinov3_weights is not None else None
        if weights is None:
            checkpoint_dir = repo / "checkpoints"
            matches = sorted(checkpoint_dir.glob("dinov3_vitl16_pretrain*.pth"))
            if len(matches) == 1:
                weights = matches[0]
            elif len(matches) > 1:
                raise FileExistsError(
                    f"Multiple DINOv3 checkpoints found under {checkpoint_dir}; pass --dinov3-weights explicitly."
                )
            else:
                raise FileNotFoundError(
                    "DINOv3 weights were not found. Put the DINOv3 ViT-L/16 checkpoint under "
                    f"{checkpoint_dir} or pass --dinov3-weights."
                )
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 weights do not exist: {weights}")

        return torch.hub.load(
            str(repo),
            self.model_name,
            source="local",
            weights=str(weights),
            trust_repo=True,
        )

    @property
    def patch_size(self) -> int:
        return int(self.spec.patch_size)

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "patch_size": int(self.patch_size),
            "output_dim": int(self.output_dim),
            "selected_layers": None if self.selected_layers is None else list(self.selected_layers),
            "device": str(self.device),
            "use_half": bool(self.use_half),
        }

    def encode_images(self, rgb_images: np.ndarray | Sequence[np.ndarray]) -> DinoFeatureMap:
        """Return a DINO feature grid with shape ``(B, patch_h, patch_w, D)``."""

        rgb_np = np.asarray(rgb_images)
        if rgb_np.ndim == 3:
            rgb_np = rgb_np[None, ...]
        if rgb_np.ndim != 4 or rgb_np.shape[-1] != 3:
            raise ValueError(f"RGB input must have shape (B,H,W,3), got {rgb_np.shape}")

        image_height = int(rgb_np.shape[1])
        image_width = int(rgb_np.shape[2])
        effective_height = (image_height // self.patch_size) * self.patch_size
        effective_width = (image_width // self.patch_size) * self.patch_size
        if effective_height <= 0 or effective_width <= 0:
            raise ValueError(
                f"Image size {image_width}x{image_height} is smaller than patch size {self.patch_size}."
            )

        rgb_np = np.ascontiguousarray(rgb_np[:, :effective_height, :effective_width, :])
        rgb = torch.as_tensor(rgb_np, device=self.device)
        rgb = rgb.permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        if self.use_half:
            rgb = rgb.half()

        with torch.no_grad():
            if self.model_name == "dummy":
                features = self._dummy_features(rgb.float())
            else:
                features = self._model_features(rgb)

        return DinoFeatureMap(
            features=features.float(),
            patch_size=int(self.patch_size),
            image_height=image_height,
            image_width=image_width,
            effective_height=effective_height,
            effective_width=effective_width,
            model_name=self.model_name,
            selected_layers=self.selected_layers,
        )

    def _model_features(self, rgb: torch.Tensor) -> torch.Tensor:
        patch_h = int(rgb.shape[-2] // self.patch_size)
        patch_w = int(rgb.shape[-1] // self.patch_size)
        n_patches = patch_h * patch_w
        layer_selector: int | list[int]
        layer_selector = list(self.selected_layers) if self.selected_layers is not None else 1

        feats = self.model.get_intermediate_layers(
            rgb,
            n=layer_selector,
            reshape=False,
            return_class_token=False,
        )
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        patch_tokens = torch.cat(list(feats), dim=-1)
        if patch_tokens.shape[1] != n_patches:
            # Some ViT variants expose register tokens before patch tokens.
            if patch_tokens.shape[1] < n_patches:
                raise RuntimeError(
                    f"DINO returned {patch_tokens.shape[1]} tokens, expected at least {n_patches}."
                )
            patch_tokens = patch_tokens[:, -n_patches:, :]
        return patch_tokens.reshape(rgb.shape[0], patch_h, patch_w, patch_tokens.shape[-1])

    def _dummy_features(self, rgb: torch.Tensor) -> torch.Tensor:
        patch_h = int(rgb.shape[-2] // self.patch_size)
        patch_w = int(rgb.shape[-1] // self.patch_size)
        pooled = F.avg_pool2d(rgb, kernel_size=self.patch_size, stride=self.patch_size)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch_h, device=rgb.device, dtype=pooled.dtype),
            torch.linspace(-1.0, 1.0, patch_w, device=rgb.device, dtype=pooled.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).expand(rgb.shape[0], -1, -1, -1)
        norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
        ones = torch.ones((rgb.shape[0], 1, patch_h, patch_w), device=rgb.device, dtype=pooled.dtype)
        features = torch.cat([pooled, coords, norm, ones, pooled[:, :1] * coords[:, :1]], dim=1)
        return features.permute(0, 2, 3, 1).contiguous()

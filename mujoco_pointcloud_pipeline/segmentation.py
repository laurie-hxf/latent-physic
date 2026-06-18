from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class InstanceMask:
    mask: np.ndarray
    label: str
    score: float = 1.0
    box_xyxy: np.ndarray | None = None


class MaskPredictor:
    def predict(self, image_rgb: np.ndarray, text_prompt: str, *, frame_index: int = 0, camera_name: str = "") -> list[InstanceMask]:
        raise NotImplementedError


class ColorThresholdMaskPredictor(MaskPredictor):
    """Synthetic-data fallback used only for smoke tests.

    This does not use MuJoCo labels. It segments saturated red/blue pixels from
    rendered RGB, which is useful for validating the RGB-D geometry path when
    GroundingDINO/SAM2 weights are unavailable.
    """

    def predict(self, image_rgb: np.ndarray, text_prompt: str, *, frame_index: int = 0, camera_name: str = "") -> list[InstanceMask]:
        image = np.asarray(image_rgb, dtype=np.uint8)
        rgb = image.astype(np.float32)
        red = (rgb[..., 0] > 100.0) & (rgb[..., 0] > 1.25 * rgb[..., 1]) & (rgb[..., 0] > 1.15 * rgb[..., 2])
        blue = (rgb[..., 2] > 100.0) & (rgb[..., 2] > 1.15 * rgb[..., 0]) & (rgb[..., 2] > 1.15 * rgb[..., 1])
        mask = red | blue
        return [InstanceMask(mask=mask, label=text_prompt, score=1.0)]


class SavedMaskPredictor(MaskPredictor):
    def __init__(self, mask_root: Path) -> None:
        self.mask_root = Path(mask_root)

    def predict(self, image_rgb: np.ndarray, text_prompt: str, *, frame_index: int = 0, camera_name: str = "") -> list[InstanceMask]:
        candidates = [
            self.mask_root / camera_name / "mask" / f"{frame_index:06d}.png",
            self.mask_root / camera_name / f"{frame_index:06d}.png",
            self.mask_root / f"{camera_name}_{frame_index:06d}.png",
            self.mask_root / f"{frame_index:06d}.png",
        ]
        mask_path = next((path for path in candidates if path.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Could not find saved mask for camera={camera_name!r}, frame={frame_index}")

        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("SavedMaskPredictor requires opencv-python to read PNG masks.") from exc

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask image: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return [InstanceMask(mask=mask > 0, label=text_prompt, score=1.0)]


class GroundedSam2MaskPredictor(MaskPredictor):
    def __init__(
        self,
        *,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint: Path | None = None,
        device: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.30,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        if sam2_checkpoint is None:
            raise ValueError("--sam2-checkpoint is required for GroundedSam2MaskPredictor.")
        self.image_predictor = SAM2ImagePredictor(build_sam2(sam2_config, str(sam2_checkpoint)))
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

    def predict(self, image_rgb: np.ndarray, text_prompt: str, *, frame_index: int = 0, camera_name: str = "") -> list[InstanceMask]:
        from PIL import Image

        image_np = np.asarray(image_rgb, dtype=np.uint8)
        image = Image.fromarray(image_np)
        prompt = text_prompt.strip()
        if prompt and not prompt.endswith("."):
            prompt = prompt + "."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )
        boxes = results[0]["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results[0]["scores"].detach().cpu().numpy().astype(np.float32)
        labels = list(results[0]["labels"])
        if len(boxes) == 0:
            return []

        self.image_predictor.set_image(image_np)
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 3 and masks.shape[0] != len(boxes):
            masks = masks[None, 0]
        elif masks.ndim == 2:
            masks = masks[None]
        if masks.shape[0] != len(boxes):
            raise RuntimeError(f"SAM2 returned {masks.shape[0]} masks for {len(boxes)} boxes.")

        instances: list[InstanceMask] = []
        for mask, label, score, box in zip(masks, labels, scores, boxes, strict=True):
            instances.append(
                InstanceMask(
                    mask=np.asarray(mask, dtype=bool),
                    label=str(label),
                    score=float(score),
                    box_xyxy=np.asarray(box, dtype=np.float32),
                )
            )
        return instances


def combine_instance_masks(instances: list[InstanceMask], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for instance in instances:
        instance_mask = np.asarray(instance.mask, dtype=bool)
        if instance_mask.shape != shape:
            raise ValueError(f"Mask shape {instance_mask.shape} does not match image shape {shape}.")
        mask |= instance_mask
    return mask

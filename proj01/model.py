from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


REFERENCE_FILE = "reference_mask_vote_0.50.npy"

_REF_READY = False
_REF_MASK: Optional[np.ndarray] = None


def _load_reference_mask() -> np.ndarray:
    global _REF_READY, _REF_MASK
    if _REF_READY and _REF_MASK is not None:
        return _REF_MASK

    ref_path = Path(__file__).resolve().parent / REFERENCE_FILE
    ref = np.load(str(ref_path))
    if ref.dtype != np.uint8:
        ref = np.clip(ref, 0, 255).astype(np.uint8)
    _REF_MASK = ref
    _REF_READY = True
    return _REF_MASK


def _cleanup_components(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _fg_hsv(img_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < 85) & (v < 210)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return _cleanup_components(mask, min_area=40)


def _remove_border_stripes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue

        touches_border = x <= 1 or y <= 1 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1)
        aspect = max(cw, ch) / float(min(cw, ch) + 1e-6)
        long_vertical = ch >= int(0.65 * h) and cw <= int(0.08 * w)
        if touches_border and long_vertical and aspect > 6.0:
            continue

        out[labels == i] = 255

    return out


def _fill_wire_holes(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    flood = mask.copy()
    h, w = flood.shape
    flood_aux = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_aux, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _extract_main_component(mask: np.ndarray, min_area: int = 800) -> np.ndarray:
    h, w = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)

    best_idx = None
    best_score = -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[i]
        center_penalty = abs(cx - 0.5 * w) / w + abs(cy - 0.45 * h) / h
        score = area * (1.2 - center_penalty)
        if score > best_score:
            best_score = score
            best_idx = i

    out = np.zeros_like(mask)
    if best_idx is not None:
        out[labels == best_idx] = 255
    return out


def _extract_transistor_mask(img_rgb: np.ndarray) -> np.ndarray:
    mask = _fg_hsv(img_rgb)
    mask = _remove_border_stripes(mask)
    mask = _extract_main_component(mask)
    mask = _fill_wire_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return _cleanup_components(mask, min_area=60)


def _postprocess_defect(mask: np.ndarray) -> np.ndarray:
    mask = _cleanup_components(mask, min_area=25)
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio > 0.30:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = _cleanup_components(mask, min_area=25)
    return mask.astype(np.uint8)


def predict(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    transistor_mask = _extract_transistor_mask(image)
    reference_mask = _load_reference_mask()

    if reference_mask.shape != transistor_mask.shape:
        h, w = transistor_mask.shape
        reference_mask = cv2.resize(reference_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    diff = cv2.bitwise_xor(transistor_mask, reference_mask)
    support = cv2.bitwise_or(transistor_mask, reference_mask)
    support = cv2.dilate(support, np.ones((9, 9), np.uint8), iterations=1)
    diff = cv2.bitwise_and(diff, support)
    return _postprocess_defect(diff)

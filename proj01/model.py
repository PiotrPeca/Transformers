from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


_REF_READY = False
_REF_SMALL: Optional[np.ndarray] = None
_REF_FULL: Optional[np.ndarray] = None


def _load_good_reference(image_shape: Tuple[int, int]) -> None:
    """Build a lightweight grayscale reference model from train/good if available."""
    global _REF_READY, _REF_SMALL, _REF_FULL
    if _REF_READY:
        return

    h, w = image_shape
    root = Path(__file__).resolve().parent / "transistor" / "train" / "good"
    if not root.exists():
        _REF_READY = True
        return

    files = sorted(root.glob("*.png"))[:80]
    if not files:
        _REF_READY = True
        return

    small_stack = []
    full_stack = []
    for file_path in files:
        bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        full_stack.append(gray.astype(np.uint8))
        small_stack.append(cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA))

    if full_stack:
        _REF_FULL = np.stack(full_stack, axis=0)
        _REF_SMALL = np.stack(small_stack, axis=0)

    _REF_READY = True


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Create broad transistor mask to suppress background false positives."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return fg


def _anomaly_map(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute anomaly score map and global mismatch score."""
    gray_f = gray.astype(np.float32)

    if _REF_FULL is not None and _REF_SMALL is not None and _REF_FULL.shape[0] >= 3:
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref_small_f = _REF_SMALL.astype(np.float32)
        dists = np.mean(np.abs(ref_small_f - small[None, :, :]), axis=(1, 2))

        k = min(5, len(dists))
        best_idx = np.argpartition(dists, k - 1)[:k]
        refs = _REF_FULL[best_idx].astype(np.float32)

        ref_median = np.median(refs, axis=0)
        ref_std = np.std(refs, axis=0)

        diff = np.abs(gray_f - ref_median)
        grad_img = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_ref = cv2.Sobel(ref_median, cv2.CV_32F, 1, 0, ksize=3)
        grad_diff = np.abs(grad_img - grad_ref)

        score = diff / (ref_std + 6.0) + 0.35 * grad_diff / 32.0
        global_mismatch = float(np.mean(dists[best_idx]))
    else:
        # Fallback when train/good is unavailable: local residual + edge response.
        smooth = cv2.GaussianBlur(gray_f, (17, 17), 0)
        residual = np.abs(gray_f - smooth)
        lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
        score = residual + 0.6 * np.abs(lap)
        global_mismatch = float(np.mean(residual))

    score = cv2.GaussianBlur(score, (5, 5), 0)
    return score, global_mismatch


def _postprocess(binary: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Clean mask while preserving very small defects."""
    mask = cv2.bitwise_and(binary, fg)

    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio > 0.20:
        # For very large anomalies (often misplaced), keep shape coherent.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    else:
        # For small defects, avoid aggressive closing that could erase thin damage.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_area = max(20, int(0.00005 * mask.size))
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == idx] = 255

    return cleaned


def predict(image: np.ndarray) -> np.ndarray:
    """Segment transistor defect mask.

    Args:
        image: RGB image of shape (H, W, 3), dtype uint8.

    Returns:
        Binary defect mask of shape (H, W), dtype uint8 with values {0, 255}.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h, w = image.shape[:2]
    _load_good_reference((h, w))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    fg = _foreground_mask(gray)
    score, global_mismatch = _anomaly_map(gray)

    fg_scores = score[fg > 0]
    if fg_scores.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Strong global mismatch is typically a misplaced transistor.
    if global_mismatch > 15.0:
        return cv2.morphologyEx(
            fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1
        ).astype(np.uint8)

    # Weak tail response on anomaly map is usually a false positive on good sample.
    q99 = float(np.percentile(fg_scores, 99.0))
    if q99 < 6.0:
        return np.zeros((h, w), dtype=np.uint8)

    threshold = float(np.percentile(fg_scores, 98.8))
    binary = (score >= threshold).astype(np.uint8) * 255
    mask = _postprocess(binary, fg)

    if float(np.count_nonzero(mask)) / float(mask.size) > 0.30:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    return mask.astype(np.uint8)

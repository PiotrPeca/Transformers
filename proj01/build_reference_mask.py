from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VOTE_THRESHOLD = 0.5
OUTPUT_FILE = "reference_mask_vote_0.50.npy"


def _load_rgb(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if target_shape is not None and rgb.shape[:2] != target_shape:
        h, w = target_shape
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb


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


def main() -> None:
    root = Path(__file__).resolve().parent
    train_good_dir = root / "transistor" / "train" / "good"
    files = sorted(train_good_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files in {train_good_dir}")

    first = _load_rgb(files[0])
    h, w = first.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)

    for file_path in files:
        img = _load_rgb(file_path, target_shape=(h, w))
        mask = _extract_transistor_mask(img)
        acc += (mask > 0).astype(np.float32)

    vote_map = acc / float(len(files))
    ref = (vote_map >= VOTE_THRESHOLD).astype(np.uint8) * 255
    ref = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    ref = cv2.dilate(ref, np.ones((9, 9), np.uint8), iterations=1)
    ref = _cleanup_components(ref, min_area=200).astype(np.uint8)

    out_path = root / OUTPUT_FILE
    np.save(str(out_path), ref)

    coverage = float(np.count_nonzero(ref)) / float(ref.size)
    print(f"Saved: {out_path}")
    print(f"Shape: {ref.shape}, dtype: {ref.dtype}, foreground_ratio: {coverage:.4f}")


if __name__ == "__main__":
    main()

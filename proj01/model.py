from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


REFERENCE_FILE = "reference_mask_vote_0.50.npy"
DEFAULT_DISTANCE_THRESHOLD_PX = 108
DEFAULT_MIN_TRANSISTOR_AREA_RATIO = 0.10
DEFAULT_CUT_LEAD_ZONE_START = 0.68
DEFAULT_CUT_LEAD_EROSION_ITERATIONS = 4
DEFAULT_CUT_LEAD_MIN_ZONE_DEFECT_AREA = 3000
DEFAULT_CUT_LEAD_MIN_COMPONENT_AREA = 25
DEFAULT_CUT_LEAD_DECISION_MODE = "any"

_REF_MASK: Optional[np.ndarray] = None


def _cleanup_components(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
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
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue

        touches_border = x <= 1 or y <= 1 or (x + comp_w) >= (w - 1) or (y + comp_h) >= (h - 1)
        aspect = max(comp_w, comp_h) / float(min(comp_w, comp_h) + 1e-6)
        long_vertical = comp_h >= int(0.65 * h) and comp_w <= int(0.08 * w)
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
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)

    best_idx = None
    best_score = -1.0
    for i in range(1, n_labels):
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


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Maska musi byc 2D.")
    return (mask > 0).astype(np.uint8) * 255


def _extract_core_mask(
    mask: np.ndarray,
    erosion_ratio: float = 0.04,
    min_kernel: int = 3,
    max_kernel: int = 31,
) -> np.ndarray:
    binary = _ensure_binary_mask(mask)
    h, w = binary.shape
    kernel_size = int(round(min(h, w) * erosion_ratio))
    kernel_size = max(min_kernel, min(max_kernel, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    core = cv2.erode(binary, kernel, iterations=1)
    if np.count_nonzero(core) == 0:
        return binary
    return core


def compute_centroid(mask: np.ndarray) -> tuple[float, float]:
    binary = _ensure_binary_mask(mask)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        raise ValueError("Maska nie zawiera obiektu foreground.")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(np.argmax(component_areas)) + 1
    cx, cy = centroids[max_idx]
    return float(cx), float(cy)


def _centroid_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def _load_reference_mask() -> np.ndarray:
    global _REF_MASK
    if _REF_MASK is None:
        mask_path = Path(__file__).resolve().parent / REFERENCE_FILE
        _REF_MASK = np.load(str(mask_path)).astype(np.uint8)
        _REF_MASK = _ensure_binary_mask(_REF_MASK)
    return _REF_MASK


def is_misplased(
    image: np.ndarray,
    distance_threshold_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    reference_mask: np.ndarray | None = None,
    precomputed_test_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> bool | tuple[bool, dict]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image musi byc typu np.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image musi miec ksztalt (H, W, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if distance_threshold_px <= 0:
        raise ValueError("distance_threshold_px musi byc dodatni")

    ref_mask = _load_reference_mask() if reference_mask is None else _ensure_binary_mask(reference_mask)

    if precomputed_test_mask is None:
        test_mask = _extract_transistor_mask(image)
    else:
        test_mask = _ensure_binary_mask(precomputed_test_mask)

    if test_mask.shape != ref_mask.shape:
        h, w = test_mask.shape
        ref_mask = cv2.resize(ref_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ref_core_mask = _extract_core_mask(ref_mask)
    test_core_mask = _extract_core_mask(test_mask)

    no_transistor_detected = np.count_nonzero(test_core_mask) == 0
    if no_transistor_detected:
        decision = True
        distance_px = float("inf")
        ref_centroid = compute_centroid(ref_core_mask)
        test_centroid = (float("nan"), float("nan"))
    else:
        ref_centroid = compute_centroid(ref_core_mask)
        test_centroid = compute_centroid(test_core_mask)
        distance_px = _centroid_distance(ref_centroid, test_centroid)
        decision = distance_px > float(distance_threshold_px)

    if not return_debug:
        return decision

    debug = {
        "distance_px": distance_px,
        "threshold_px": float(distance_threshold_px),
        "ref_centroid": ref_centroid,
        "test_centroid": test_centroid,
        "ref_mask": ref_mask,
        "test_mask": test_mask,
        "ref_core_mask": ref_core_mask,
        "test_core_mask": test_core_mask,
        "no_transistor_detected": no_transistor_detected,
    }
    return decision, debug


is_misplaced = is_misplased


def is_misplased_with_presence_check(
    image: np.ndarray,
    distance_threshold_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    min_area_ratio: float = DEFAULT_MIN_TRANSISTOR_AREA_RATIO,
    reference_mask: np.ndarray | None = None,
    precomputed_test_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> bool | tuple[bool, dict]:
    if min_area_ratio <= 0 or min_area_ratio >= 1:
        raise ValueError("min_area_ratio musi byc w zakresie (0, 1)")

    ref_mask = _load_reference_mask() if reference_mask is None else _ensure_binary_mask(reference_mask)

    if precomputed_test_mask is None:
        test_mask = _extract_transistor_mask(image)
    else:
        test_mask = _ensure_binary_mask(precomputed_test_mask)

    if test_mask.shape != ref_mask.shape:
        h, w = test_mask.shape
        ref_mask = cv2.resize(ref_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ref_area = int(np.count_nonzero(ref_mask))
    test_area = int(np.count_nonzero(test_mask))
    area_limit = max(int(round(ref_area * float(min_area_ratio))), 1)

    if test_area < area_limit:
        decision = True
        debug = {
            "decision_reason": "area_check",
            "distance_px": float("inf"),
            "threshold_px": float(distance_threshold_px),
            "ref_area": ref_area,
            "test_area": test_area,
            "area_limit": area_limit,
            "min_area_ratio": float(min_area_ratio),
            "ref_mask": ref_mask,
            "test_mask": test_mask,
        }
        return (decision, debug) if return_debug else decision

    decision, base_debug = is_misplased(
        image=image,
        distance_threshold_px=distance_threshold_px,
        reference_mask=ref_mask,
        precomputed_test_mask=test_mask,
        return_debug=True,
    )
    base_debug.update(
        {
            "decision_reason": "centroid_shift",
            "ref_area": ref_area,
            "test_area": test_area,
            "area_limit": area_limit,
            "min_area_ratio": float(min_area_ratio),
        }
    )
    return (decision, base_debug) if return_debug else decision


def _build_misplaced_mask(debug: dict, fallback_shape: Tuple[int, int]) -> np.ndarray:
    ref_mask = debug.get("ref_mask")
    test_mask = debug.get("test_mask")
    if not isinstance(ref_mask, np.ndarray) or not isinstance(test_mask, np.ndarray):
        return np.zeros(fallback_shape, dtype=np.uint8)

    ref_mask = _ensure_binary_mask(ref_mask)
    test_mask = _ensure_binary_mask(test_mask)
    if ref_mask.shape != test_mask.shape:
        h, w = test_mask.shape
        ref_mask = cv2.resize(ref_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    misplaced_mask = cv2.bitwise_xor(ref_mask, test_mask)
    if np.count_nonzero(misplaced_mask) == 0:
        misplaced_mask = ref_mask.copy()
    return misplaced_mask


def is_cut_lead(
    image: np.ndarray,
    reference_mask: np.ndarray | None = None,
    lead_zone_start: float = DEFAULT_CUT_LEAD_ZONE_START,
    erosion_iterations: int = DEFAULT_CUT_LEAD_EROSION_ITERATIONS,
    min_zone_defect_area: int = DEFAULT_CUT_LEAD_MIN_ZONE_DEFECT_AREA,
    min_component_area: int = DEFAULT_CUT_LEAD_MIN_COMPONENT_AREA,
    decision_mode: str = DEFAULT_CUT_LEAD_DECISION_MODE,
    thresholds_by_zone: dict[str, int] | None = None,
    precomputed_test_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> bool | tuple[bool, dict]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image musi byc typu np.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image musi miec ksztalt (H, W, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if not (0.0 < float(lead_zone_start) < 1.0):
        raise ValueError("lead_zone_start musi byc w zakresie (0, 1)")
    if int(erosion_iterations) < 0:
        raise ValueError("erosion_iterations musi byc >= 0")
    if int(min_zone_defect_area) < 0:
        raise ValueError("min_zone_defect_area musi byc >= 0")
    if int(min_component_area) < 0:
        raise ValueError("min_component_area musi byc >= 0")

    ref_mask = _load_reference_mask() if reference_mask is None else _ensure_binary_mask(reference_mask)
    if precomputed_test_mask is None:
        test_mask = _extract_transistor_mask(image)
    else:
        test_mask = _ensure_binary_mask(precomputed_test_mask)

    if test_mask.shape != ref_mask.shape:
        h_ref, w_ref = ref_mask.shape
        test_mask = cv2.resize(test_mask, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST)

    h, w = ref_mask.shape
    roi_h = int(h * 0.5)
    ref_top = ref_mask[:roi_h, :]
    test_top = test_mask[:roi_h, :]
    m_ref = cv2.moments(ref_top)
    m_test = cv2.moments(test_top)

    if m_ref["m00"] == 0 or m_test["m00"] == 0:
        dx, dy = 0, 0
    else:
        dx = int(m_ref["m10"] / m_ref["m00"] - m_test["m10"] / m_test["m00"])
        dy = int(m_ref["m01"] / m_ref["m00"] - m_test["m01"] / m_test["m00"])

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_mask = cv2.warpAffine(test_mask, M, (w, h), flags=cv2.INTER_NEAREST)

    ys, xs = np.where(ref_mask > 0)
    if xs.size == 0:
        raise ValueError("Maska referencyjna jest pusta.")

    y0 = int(np.clip(int(h * float(lead_zone_start)), 0, h - 1))
    x_min = int(xs.min())
    x_max = int(xs.max())
    if x_max <= x_min:
        raise ValueError("Niepoprawny zakres X maski referencyjnej.")

    width = x_max - x_min + 1
    t1 = x_min + width // 3
    t2 = x_min + (2 * width) // 3
    zones = {
        "left": (int(np.clip(x_min, 0, w)), int(np.clip(t1, 0, w))),
        "center": (int(np.clip(t1, 0, w)), int(np.clip(t2, 0, w))),
        "right": (int(np.clip(t2, 0, w)), int(np.clip(x_max + 1, 0, w))),
    }

    for name, (xa, xb) in list(zones.items()):
        if xb <= xa:
            xb = min(w, xa + 1)
        zones[name] = (xa, xb)

    kernel = np.ones((3, 3), np.uint8)
    ref_leads = ref_mask[y0:, :]
    test_leads = aligned_mask[y0:, :]
    ref_eroded = cv2.erode(ref_leads, kernel, iterations=int(erosion_iterations))
    diff_full = cv2.subtract(ref_eroded, test_leads)
    diff_full = cv2.morphologyEx(diff_full, cv2.MORPH_OPEN, kernel)
    diff_full = _cleanup_components(diff_full, min_area=int(min_component_area))

    zone_pixels: dict[str, int] = {}
    zone_thresholds: dict[str, int] = {}
    zone_flags: dict[str, bool] = {}
    for zone_name, (x0, x1) in zones.items():
        zone_view = diff_full[:, x0:x1]
        pixels = int(np.count_nonzero(zone_view))
        if thresholds_by_zone is None:
            threshold = int(min_zone_defect_area)
        else:
            threshold = int(thresholds_by_zone[zone_name])
        zone_pixels[zone_name] = pixels
        zone_thresholds[zone_name] = threshold
        zone_flags[zone_name] = pixels > threshold

    if decision_mode == "any":
        is_defective = bool(zone_flags["left"] or zone_flags["center"] or zone_flags["right"])
    elif decision_mode == "majority":
        is_defective = int(zone_flags["left"]) + int(zone_flags["center"]) + int(zone_flags["right"]) >= 2
    else:
        raise ValueError(f"Nieobslugiwany decision_mode: {decision_mode}")

    if not return_debug:
        return is_defective

    diff_mask_full = np.zeros_like(ref_mask)
    diff_mask_full[y0:, :] = diff_full
    debug = {
        "is_defective": bool(is_defective),
        "zone_pixels": zone_pixels,
        "zone_flags": zone_flags,
        "zone_thresholds": zone_thresholds,
        "zones": zones,
        "y0": y0,
        "dx": int(dx),
        "dy": int(dy),
        "ref_mask": ref_mask,
        "test_mask": test_mask,
        "aligned_mask": aligned_mask,
        "diff_mask": diff_mask_full,
    }
    return is_defective, debug


def predict(image: np.ndarray) -> Tuple[np.ndarray, str]:
    decision, debug = is_misplased_with_presence_check(
        image=image,
        distance_threshold_px=DEFAULT_DISTANCE_THRESHOLD_PX,
        min_area_ratio=DEFAULT_MIN_TRANSISTOR_AREA_RATIO,
        reference_mask=_load_reference_mask(),
        return_debug=True,
    )

    h, w = image.shape[:2]
    if decision:
        return _build_misplaced_mask(debug, fallback_shape=(h, w)), "misplaced"

    return np.zeros((h, w), dtype=np.uint8), "good"


def predict_debug(image: np.ndarray) -> Tuple[np.ndarray, str]:
    return predict(image)
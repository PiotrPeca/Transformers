from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

import Transformers.proj01.model as model


CLASSES = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]
REFERENCE_FILE = "reference_mask_vote_0.50.npy"


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def iter_test_images(test_dir: Path) -> Iterable[Tuple[str, Path]]:
    for cls_name in CLASSES:
        for img_path in sorted((test_dir / cls_name).glob("*.png")):
            yield cls_name, img_path


def load_gt_or_zero(gt_dir: Path, cls_name: str, img_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if cls_name == "good":
        return np.zeros(shape, dtype=np.uint8)
    gt_path = gt_dir / cls_name / f"{img_path.stem}_mask.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Missing GT mask: {gt_path}")
    if gt.shape != shape:
        h, w = shape
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    return (gt > 0).astype(np.uint8) * 255


def evaluate_reference_iou(project_dir: Path) -> Dict[str, float]:
    test_dir = project_dir / "transistor" / "test"
    ref_path = project_dir / REFERENCE_FILE
    reference = np.load(str(ref_path)).astype(np.uint8)

    per_class: Dict[str, list[float]] = {name: [] for name in CLASSES}
    all_scores = []

    print("=== Reference IoU (extract_mask vs reference_mask) ===")
    for cls_name, img_path in iter_test_images(test_dir):
        image = load_rgb(img_path)
        extracted = model._extract_transistor_mask(image)

        if extracted.shape != reference.shape:
            h, w = extracted.shape
            ref_resized = cv2.resize(reference, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            ref_resized = reference

        score = iou(extracted, ref_resized)
        per_class[cls_name].append(score)
        all_scores.append(score)
        print(f"{cls_name:12s} {img_path.name:12s} ref_IoU={score:.4f}")

    summary: Dict[str, float] = {}
    print("\n--- Summary (reference IoU) ---")
    for cls_name in CLASSES:
        arr = np.array(per_class[cls_name], dtype=np.float32)
        mean_value = float(arr.mean()) if arr.size else float("nan")
        summary[f"ref_iou_{cls_name}"] = mean_value
        print(f"{cls_name:12s} mean={mean_value:.6f} n={arr.size}")

    all_arr = np.array(all_scores, dtype=np.float32)
    overall = float(all_arr.mean()) if all_arr.size else float("nan")
    summary["ref_iou_overall"] = overall
    print(f"overall      mean={overall:.6f}")
    return summary


def evaluate_competition_iou(project_dir: Path) -> Dict[str, float]:
    test_dir = project_dir / "transistor" / "test"
    gt_dir = project_dir / "transistor" / "ground_truth"

    per_class: Dict[str, list[float]] = {name: [] for name in CLASSES}
    all_scores = []

    print("\n=== Competition-like IoU (predict vs GT) ===")
    for cls_name, img_path in iter_test_images(test_dir):
        image = load_rgb(img_path)
        pred = model.predict(image)
        gt = load_gt_or_zero(gt_dir, cls_name, img_path, image.shape[:2])

        score = iou(pred, gt)
        per_class[cls_name].append(score)
        all_scores.append(score)
        print(f"{cls_name:12s} {img_path.name:12s} comp_IoU={score:.4f}")

    summary: Dict[str, float] = {}
    print("\n--- Summary (competition IoU) ---")
    for cls_name in CLASSES:
        arr = np.array(per_class[cls_name], dtype=np.float32)
        mean_value = float(arr.mean()) if arr.size else float("nan")
        summary[f"comp_iou_{cls_name}"] = mean_value
        print(f"{cls_name:12s} mean={mean_value:.6f} n={arr.size}")

    all_arr = np.array(all_scores, dtype=np.float32)
    overall = float(all_arr.mean()) if all_arr.size else float("nan")
    summary["comp_iou_overall"] = overall
    print(f"overall      mean={overall:.6f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local checker for reference and competition-style IoU.")
    parser.add_argument(
        "--mode",
        choices=["reference", "competition", "both"],
        default="both",
        help="Which metric mode to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    if args.mode in {"reference", "both"}:
        evaluate_reference_iou(project_dir)
    if args.mode in {"competition", "both"}:
        evaluate_competition_iou(project_dir)


if __name__ == "__main__":
    main()

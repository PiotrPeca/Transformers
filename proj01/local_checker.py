from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple
import cv2
import numpy as np
import model

CLASSES = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]
REFERENCE_FILE = "reference_mask_vote_0.50.npy"

def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0: return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)

def iter_test_images(test_dir: Path) -> Iterable[Tuple[str, Path]]:
    for cls_name in CLASSES:
        for img_path in sorted((test_dir / cls_name).glob("*.png")):
            yield cls_name, img_path

def load_gt_or_zero(gt_dir: Path, cls_name: str, img_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if cls_name == "good": return np.zeros(shape, dtype=np.uint8)
    gt_path = gt_dir / cls_name / f"{img_path.stem}_mask.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None: raise FileNotFoundError(f"Missing: {gt_path}")
    if gt.shape != shape: gt = cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (gt > 0).astype(np.uint8) * 255

def evaluate_competition_iou(project_dir: Path) -> Dict[str, float]:
    test_dir = project_dir / "transistor" / "test"
    gt_dir = project_dir / "transistor" / "ground_truth"
    per_class = {name: [] for name in CLASSES}
    all_scores = []
    
    # Confusion matrix tracker
    confusion = {tc: {pc: 0 for pc in CLASSES} for tc in CLASSES}

    print("\n=== Competition-like IoU (predict vs GT) ===")
    for cls_name, img_path in iter_test_images(test_dir):
        image = load_rgb(img_path)
        
        # Use predict_debug to get the label
        pred, pred_lbl = model.predict_debug(image)
        
        gt = load_gt_or_zero(gt_dir, cls_name, img_path, image.shape[:2])
        score = iou(pred, gt)
        per_class[cls_name].append(score)
        all_scores.append(score)
        confusion[cls_name][pred_lbl] += 1
        
        print(f"{cls_name:12s} {img_path.name:12s} comp_IoU={score:.4f} pred=[{pred_lbl}]")

    print("\n--- Predictions (True -> Predicted) ---")
    for tc in CLASSES:
        preds_str = ", ".join(f"{pc}: {confusion[tc][pc]}" for pc in CLASSES if confusion[tc][pc] > 0)
        print(f"True {tc:12s} -> {preds_str}")

    summary = {}
    print("\n--- Summary (competition IoU) ---")
    for cls_name in CLASSES:
        arr = np.array(per_class[cls_name], dtype=np.float32)
        mean_value = float(arr.mean()) if arr.size else float("nan")
        summary[f"comp_iou_{cls_name}"] = mean_value
        print(f"{cls_name:12s} mean={mean_value:.6f} n={arr.size}")

    overall = np.array(all_scores, dtype=np.float32).mean()
    print(f"overall      mean={overall:.6f}")
    return summary

def main():
    evaluate_competition_iou(Path(__file__).resolve().parent)

if __name__ == "__main__":
    main()

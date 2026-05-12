from pathlib import Path
import cv2
import model

project_dir = Path.cwd()
test_dir = project_dir / "transistor" / "test"

def load_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

good_files = sorted((test_dir / "good").glob("*.png"))
cut_files = sorted((test_dir / "cut_lead").glob("*.png"))

lead_zone_candidates = [0.65, 0.68, 0.72, 0.75, 0.78]
erosion_candidates = [3, 4, 5, 6]
threshold_candidates = [3000, 4500, 6500, 8500, 10000]
component_candidates = [25, 40, 60]
decision_modes = ["any", "majority"]

cache = {p: load_rgb(p) for p in good_files + cut_files}
results = []
for lz in lead_zone_candidates:
    for er in erosion_candidates:
        for thr in threshold_candidates:
            for comp in component_candidates:
                for mode in decision_modes:
                    tp = tn = fp = fn = 0
                    for p in good_files:
                        pred = bool(model.is_cut_lead(cache[p], lead_zone_start=lz, erosion_iterations=er, min_zone_defect_area=thr, min_component_area=comp, decision_mode=mode))
                        if pred:
                            fp += 1
                        else:
                            tn += 1
                    for p in cut_files:
                        pred = bool(model.is_cut_lead(cache[p], lead_zone_start=lz, erosion_iterations=er, min_zone_defect_area=thr, min_component_area=comp, decision_mode=mode))
                        if pred:
                            tp += 1
                        else:
                            fn += 1
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                    results.append({
                        "lead_zone_start": lz,
                        "erosion_iterations": er,
                        "min_zone_defect_area": thr,
                        "min_component_area": comp,
                        "decision_mode": mode,
                        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                        "precision": precision, "recall": recall, "f1": f1,
                    })

# Priorytet: minimalizuj FP, potem FN, potem maksymalizuj TP
results.sort(key=lambda r: (r["fp"], r["fn"], -r["tp"], -r["tn"]))
print(f"good={len(good_files)} cut={len(cut_files)} combos={len(results)}")
print("TOP 15 FP-first:")
for r in results[:15]:
    print(r)
print("\nBEST_FP_FIRST:")
print(results[0])

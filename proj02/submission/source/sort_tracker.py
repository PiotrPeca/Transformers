import argparse
import os
from pathlib import Path

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Configuration (can be overridden by CLI arguments)
SEQUENCE_NAME = "MOT_07"
DATASET_SPLIT = "test"  # "train" or "test"
MIN_CONFIDENCE = 0.0
MIN_IOU = 0.3
MAX_AGE = 3


def iou_batch(bboxes1, bboxes2):
    """Compute IoU matrix between two sets of [x, y, w, h] boxes."""
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    b1 = np.asarray(bboxes1, dtype=np.float32)
    b2 = np.asarray(bboxes2, dtype=np.float32)

    b1_x1 = b1[:, 0]
    b1_y1 = b1[:, 1]
    b1_x2 = b1[:, 0] + b1[:, 2]
    b1_y2 = b1[:, 1] + b1[:, 3]

    b2_x1 = b2[:, 0]
    b2_y1 = b2[:, 1]
    b2_x2 = b2[:, 0] + b2[:, 2]
    b2_y2 = b2[:, 1] + b2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def convert_bbox_to_z(bbox):
    """Convert [x, y, w, h] to [cx, cy, s, r]."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x):
    """Convert [cx, cy, s, r] to [x, y, w, h]."""
    x = np.asarray(x).reshape(-1)
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    return np.array([x1, y1, w, h], dtype=np.float32)


class KalmanBoxTracker:
    """Represents a single tracked object with a Kalman filter."""

    _count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.id = KalmanBoxTracker._count + 1
        KalmanBoxTracker._count += 1

        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


class Tracker:
    """Manages multiple active tracks."""

    def __init__(self, min_iou, max_age):
        self.min_iou = min_iou
        self.max_age = max_age
        self.tracks = []

    def predict(self):
        for track in self.tracks:
            track.predict()

    def compute_cost_matrix(self, tracks, detections):
        track_boxes = [t.get_state() for t in tracks]
        iou_mat = iou_batch(track_boxes, detections)
        return 1.0 - iou_mat

    def update(self, detections):
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(KalmanBoxTracker(det))
            return

        if len(detections) == 0:
            self._remove_dead_tracks()
            return

        cost_matrix = self.compute_cost_matrix(self.tracks, detections)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_idx, col_idx):
            if (1.0 - cost_matrix[r, c]) < self.min_iou:
                continue
            self.tracks[r].update(detections[c])
            matched_tracks.add(r)
            matched_dets.add(c)

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(KalmanBoxTracker(det))

        self._remove_dead_tracks()

    def _remove_dead_tracks(self):
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def get_active_tracks(self):
        return [t for t in self.tracks if t.time_since_update == 0]


def load_detections(det_path, min_confidence):
    data = np.loadtxt(det_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    frames = data[:, 0].astype(int)
    confidences = data[:, 6]
    mask = confidences >= min_confidence

    data = data[mask]
    frames = frames[mask]

    return data, frames


def write_outputs(output_path, results):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.3f},{row[3]:.3f},{row[4]:.3f},{row[5]:.3f},1,-1,-1,-1\n"
            )


def run_sequence(
    dataset_root,
    sequence_name,
    dataset_split,
    min_confidence,
    min_iou,
    max_age,
    output_dir,
):
    det_path = (
        Path(dataset_root)
        / f"evs_mot-{dataset_split}/{sequence_name}"
        / "det/det.txt"
    )
    data, frames = load_detections(det_path, min_confidence)
    max_frame = int(frames.max())

    tracker = Tracker(min_iou=min_iou, max_age=max_age)
    results = []

    for frame_id in range(1, max_frame + 1):
        frame_mask = frames == frame_id
        frame_dets = data[frame_mask][:, 2:6]

        tracker.predict()
        tracker.update(frame_dets)

        for track in tracker.get_active_tracks():
            bbox = track.get_state()
            results.append(
                [frame_id, track.id, bbox[0], bbox[1], bbox[2], bbox[3]]
            )

    output_path = Path(output_dir) / f"{sequence_name}.txt"
    os.makedirs(output_path.parent, exist_ok=True)
    write_outputs(output_path, results)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SORT tracking on MOT detections")
    parser.add_argument(
        "--sequence",
        type=str,
        default=SEQUENCE_NAME,
        help="Sequence name, e.g. MOT_01",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "evs_mot_public_dataset"),
        help="Path to evs_mot_public_dataset",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DATASET_SPLIT,
        choices=["train", "test"],
        help="Dataset split to read detections from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Output directory for MOT results",
    )
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--min-iou", type=float, default=MIN_IOU)
    parser.add_argument("--max-age", type=int, default=MAX_AGE)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sequence(
        dataset_root=args.dataset_root,
        sequence_name=args.sequence,
        dataset_split=args.dataset_split,
        min_confidence=args.min_confidence,
        min_iou=args.min_iou,
        max_age=args.max_age,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

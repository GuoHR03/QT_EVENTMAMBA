import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.realtime_inference import EllipsePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read an AEDAT/AEDAT4 file, predict pupil ellipse geometry with EventMamba-v3, and visualize results."
    )
    parser.add_argument("aedat_path", help="Input .aedat or .aedat4 file.")
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "checkpoint" / "v14_new" / "P3best_checkpoint.pth"),
        help="EventMamba-v3 ellipse checkpoint. matrix_A.pt must be in the same directory.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "aedat_eventmamba_v3"))
    parser.add_argument("--window-ms", type=float, default=20.0, help="Event accumulation window for each prediction.")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of events sampled for one model input.")
    parser.add_argument("--min-events", type=int, default=1024, help="Skip windows with fewer normalized events.")
    parser.add_argument("--max-windows", type=int, default=0, help="Limit processed windows; 0 means all.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Optional ROI in source pixels.")
    parser.add_argument("--src-width", type=int, default=0, help="Override source sensor width.")
    parser.add_argument("--src-height", type=int, default=0, help="Override source sensor height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video fps.")
    parser.add_argument("--no-video", action="store_true", help="Only write CSV and preview PNGs.")
    parser.add_argument("--preview-count", type=int, default=12, help="Number of PNG preview frames to save.")
    return parser.parse_args()


def load_aedat_events(path):
    try:
        import dv_processing as dv
    except Exception as exc:
        raise RuntimeError(
            "Reading AEDAT/AEDAT4 requires dv_processing. Install it in the active Python environment first."
        ) from exc

    try:
        reader = dv.io.MonoCameraRecording(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not open AEDAT file with dv_processing: {path}") from exc

    width, height = 640, 480
    try:
        resolution = reader.getEventResolution()
        width, height = int(resolution.width), int(resolution.height)
    except Exception:
        pass

    batches = []
    read_methods = ("getNextEventBatch", "readNextEventBatch")
    while True:
        batch = None
        for method_name in read_methods:
            method = getattr(reader, method_name, None)
            if method is None:
                continue
            try:
                batch = method()
                break
            except TypeError:
                continue
        if batch is None:
            break

        events = np.asarray(batch.numpy() if hasattr(batch, "numpy") else batch)
        if len(events) == 0:
            continue
        batches.append(to_plain_events(events))

    if not batches:
        raise RuntimeError(f"No polarity events were found in {path}.")

    events = np.concatenate(batches)
    order = np.argsort(events[:, 2], kind="mergesort")
    return events[order], width, height


def to_plain_events(events):
    names = events.dtype.names or ()
    if not {"x", "y"}.issubset(names):
        raise ValueError("Event batch must contain x and y fields.")

    time_field = next((name for name in ("t", "timestamp", "ts") if name in names), None)
    if time_field is None:
        raise ValueError("Event batch must contain a timestamp field named t, timestamp, or ts.")

    return np.column_stack((events["x"], events["y"], events[time_field])).astype(np.float32)


def downsample_roi_normalize_events(data_numpy, roi, src_width=640, src_height=480):
    if data_numpy is None or len(data_numpy) == 0 or not roi:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    roi_x, roi_y, roi_width, roi_height = roi
    mask = (
        (data_numpy[:, 0] >= roi_x) & (data_numpy[:, 0] < roi_x + roi_width) &
        (data_numpy[:, 1] >= roi_y) & (data_numpy[:, 1] < roi_y + roi_height)
    )
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    cropped = data_numpy[mask]
    x_values = (cropped[:, 0] - roi_x) / roi_width
    y_values = (cropped[:, 1] - roi_y) / roi_height
    t_values = cropped[:, 2]
    x_values = np.clip(x_values, 0.0, 1.0)
    y_values = np.clip(y_values, 0.0, 1.0)

    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downsample_crop_normalize_events(data_numpy, src_width=640, src_height=480, dst_width=512, dst_height=512):
    if data_numpy is None or len(data_numpy) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    mask = (x_raw >= 96) & (x_raw <= 608)
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]

    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def iter_event_windows(events, window_us):
    start = float(events[0, 2])
    end = float(events[-1, 2])
    left = 0
    current = start
    while current <= end:
        right_time = current + window_us
        right = int(np.searchsorted(events[:, 2], right_time, side="left"))
        if right > left:
            yield current, right_time, events[left:right]
        current = right_time
        left = right


def normalize_for_model(window_events, args, src_width, src_height):
    if args.roi:
        x_norm, y_norm, t_norm = downsample_roi_normalize_events(
            window_events, tuple(args.roi), src_width=src_width, src_height=src_height
        )
    else:
        x_norm, y_norm, t_norm = downsample_crop_normalize_events(
            window_events, src_width=src_width, src_height=src_height
        )
    if len(x_norm) == 0:
        return None
    return np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)


def sample_points(points, num_points, rng):
    if len(points) < num_points:
        return None
    if len(points) == num_points:
        return points
    indices = rng.choice(len(points), num_points, replace=False)
    indices.sort()
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def draw_events_and_ellipse(frame_events, prediction, args, src_width, src_height, width, height):
    frame = np.full((height, width, 3), (30, 37, 52), dtype=np.uint8)
    if len(frame_events) > 0:
        x = np.clip(frame_events[:, 0].astype(np.int32), 0, width - 1)
        y = np.clip(frame_events[:, 1].astype(np.int32), 0, height - 1)
        frame[y, x] = (230, 235, 245)

    cx, cy, a, b, angle = [float(v) for v in prediction]
    if args.roi:
        roi_x, roi_y, roi_w, roi_h = args.roi
        center = (int(roi_x + cx * roi_w), int(roi_y + cy * roi_h))
        axes = (max(1, int(a * roi_w)), max(1, int(b * roi_h)))
    else:
        # Matches downsample_crop_normalize_events: crop x [96, 608], y [-16, 496].
        center = (int((96.0 + cx * 512.0) * src_width / 640.0), int((cy * 512.0 - 16.0) * src_height / 480.0))
        axes = (max(1, int(a * 512.0 * src_width / 640.0)), max(1, int(b * 512.0 * src_height / 480.0)))

    cv2.ellipse(frame, center, axes, np.degrees(angle), 0, 360, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 3, (0, 90, 255), -1, cv2.LINE_AA)
    text = f"x={cx:.3f} y={cy:.3f} a={a:.3f} b={b:.3f} angle={angle:.3f}"
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aedat_path = Path(args.aedat_path)
    events, detected_width, detected_height = load_aedat_events(aedat_path)
    src_width = args.src_width or detected_width
    src_height = args.src_height or detected_height

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    predictor = EllipsePredictor(args.weights, device)

    csv_path = output_dir / f"{aedat_path.stem}_eventmamba_v3_predictions.csv"
    video_path = output_dir / f"{aedat_path.stem}_eventmamba_v3_overlay.mp4"
    preview_dir = output_dir / f"{aedat_path.stem}_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (src_width, src_height))

    rows = []
    preview_stride = None
    if args.preview_count > 0:
        estimated_windows = max(1, int((events[-1, 2] - events[0, 2]) / (args.window_ms * 1000.0)))
        preview_stride = max(1, estimated_windows // args.preview_count)

    processed = 0
    for index, (start_us, end_us, window_events) in enumerate(iter_event_windows(events, args.window_ms * 1000.0)):
        normalized = normalize_for_model(window_events, args, src_width, src_height)
        if normalized is None or len(normalized) < args.min_events:
            continue
        sample = sample_points(normalized, args.num_points, rng)
        if sample is None:
            continue

        prediction = predictor.predict(sample)
        rows.append([index, int(start_us), int(end_us), len(window_events), *[float(v) for v in prediction]])

        frame = draw_events_and_ellipse(window_events, prediction, args, src_width, src_height, src_width, src_height)
        if video_writer is not None:
            video_writer.write(frame)
        if preview_stride and index % preview_stride == 0 and len(list(preview_dir.glob("*.png"))) < args.preview_count:
            cv2.imwrite(str(preview_dir / f"frame_{index:06d}.png"), frame)

        processed += 1
        if args.max_windows > 0 and processed >= args.max_windows:
            break

    if video_writer is not None:
        video_writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window_index", "start_us", "end_us", "event_count", "x", "y", "a", "b", "angle"])
        writer.writerows(rows)

    print(f"Processed windows: {processed}")
    print(f"CSV: {csv_path}")
    if video_writer is not None:
        print(f"Video: {video_path}")
    print(f"Previews: {preview_dir}")


if __name__ == "__main__":
    main()

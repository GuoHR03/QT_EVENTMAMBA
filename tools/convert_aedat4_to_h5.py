import argparse
import os
import time

import h5py

from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.event_processing import EVENT_CD_DTYPE, to_event_cd
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH


def aedat4_resolution(reader):
    try:
        resolution = reader.getEventResolution()
        if isinstance(resolution, tuple):
            return int(resolution[0]), int(resolution[1])
        return int(resolution.width), int(resolution.height)
    except Exception:
        return DEFAULT_SENSOR_WIDTH, DEFAULT_SENSOR_HEIGHT


def count_aedat4_events(input_path):
    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(str(input_path))
    total = 0
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            break
        if not batch.isEmpty():
            total += len(batch.numpy())
    return total


def convert_aedat4_to_h5(input_path, output_path, compression="gzip", progress_every=100):
    import dv_processing as dv

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_events = count_aedat4_events(input_path)
    reader = dv.io.MonoCameraRecording(str(input_path))
    width, height = aedat4_resolution(reader)

    compression_arg = None if compression == "none" else compression
    written = 0
    batches = 0
    start_time = time.perf_counter()

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["width"] = width
        h5_file.attrs["height"] = height
        h5_file.attrs["source"] = str(input_path)
        h5_file.attrs["format"] = "EVENT_CD_DTYPE"
        dataset = h5_file.create_dataset(
            "events",
            shape=(total_events,),
            maxshape=(total_events,),
            dtype=EVENT_CD_DTYPE,
            compression=compression_arg,
            chunks=True,
        )

        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None:
                break
            if batch.isEmpty():
                continue

            events = to_event_cd(batch.numpy())
            end = written + len(events)
            dataset[written:end] = events
            written = end
            batches += 1

            if progress_every and batches % progress_every == 0:
                elapsed = time.perf_counter() - start_time
                print(f"converted {written}/{total_events} events in {elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - start_time
    return {
        "input": str(input_path),
        "output": str(output_path),
        "width": width,
        "height": height,
        "events": written,
        "batches": batches,
        "seconds": elapsed,
        "size_mb": os.path.getsize(output_path) / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert AEDAT4 events to the H5 format used by this project.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    result = convert_aedat4_to_h5(
        args.input,
        args.output,
        compression=args.compression,
        progress_every=args.progress_every,
    )
    print("conversion complete")
    print(f"input: {result['input']}")
    print(f"output: {result['output']}")
    print(f"resolution: {result['width']}x{result['height']}")
    print(f"events: {result['events']}")
    print(f"batches: {result['batches']}")
    print(f"seconds: {result['seconds']:.2f}")
    print(f"output_size_mb: {result['size_mb']:.2f}")


if __name__ == "__main__":
    main()

"""Extract one real 20 ms inference sample from a Metavision RAW file."""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.event_processing import build_inference_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("output")
    parser.add_argument("--interval-us", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from metavision_core.event_io import EventsIterator

    iterator = EventsIterator(input_path=args.raw, delta_t=args.interval_us)
    height, width = iterator.get_size()
    rng = np.random.default_rng(args.seed)
    inspected = 0
    for events in iterator:
        inspected += 1
        payload = build_inference_payload(
            events,
            width=width,
            height=height,
            fallback_normalization="crop",
            rng=rng,
        )
        if payload is None:
            continue
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            events=payload["data"],
            timestamp=np.asarray(payload["timestamp"], dtype=np.int64),
            sensor_size=np.asarray([width, height], dtype=np.int64),
        )
        print(json.dumps({
            "status": "verified",
            "raw": str(Path(args.raw).resolve()),
            "sample": str(output.resolve()),
            "windows_inspected": inspected,
            "sensor_size": [width, height],
            "event_shape": list(payload["data"].shape),
            "timestamp": payload["timestamp"],
        }, indent=2))
        return
    raise RuntimeError("No inference window contained at least 1024 usable events")


if __name__ == "__main__":
    main()

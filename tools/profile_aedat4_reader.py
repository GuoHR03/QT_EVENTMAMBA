import argparse
import statistics
import time


def percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def print_stats(name, values):
    if not values:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: count={len(values)} "
        f"avg={statistics.mean(values):.3f}ms "
        f"p50={percentile(values, 50):.3f}ms "
        f"p95={percentile(values, 95):.3f}ms "
        f"max={max(values):.3f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Profile dv_processing AEDAT4 batch read and numpy conversion speed.")
    parser.add_argument("path")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(args.path)
    read_ms = []
    numpy_ms = []
    event_counts = []
    total_events = 0

    for _ in range(args.max_batches):
        if not reader.isRunning():
            break

        start = time.perf_counter()
        batch = reader.getNextEventBatch()
        read_ms.append((time.perf_counter() - start) * 1000.0)

        if batch is None:
            break
        if batch.isEmpty():
            event_counts.append(0)
            continue

        start = time.perf_counter()
        events = batch.numpy()
        numpy_ms.append((time.perf_counter() - start) * 1000.0)

        event_counts.append(len(events))
        total_events += len(events)

    print(f"file: {args.path}")
    print(f"batches: {len(event_counts)}")
    print(f"total_events: {total_events}")
    if event_counts:
        print(
            f"events_per_batch: avg={statistics.mean(event_counts):.1f} "
            f"p95={percentile(event_counts, 95):.0f} max={max(event_counts)}"
        )
    print_stats("getNextEventBatch", read_ms)
    print_stats("batch.numpy", numpy_ms)


if __name__ == "__main__":
    main()

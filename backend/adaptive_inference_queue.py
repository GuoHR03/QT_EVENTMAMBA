import queue
import time

from backend.event_pipeline import InferenceWindow


class AdaptiveInferenceQueueConsumer:
    """Select fresh inference windows without penalizing healthy queues."""

    def __init__(
        self,
        target_queue,
        stop_signal,
        clock=None,
        minimum_freshness_s=0.05,
        ewma_alpha=0.2,
    ):
        self.target_queue = target_queue
        self.stop_signal = stop_signal
        self.clock = clock or time.perf_counter
        self.minimum_freshness_s = max(0.001, float(minimum_freshness_s))
        self.ewma_alpha = min(1.0, max(0.01, float(ewma_alpha)))
        self.processing_ewma_s = None
        self.arrival_ewma_s = None
        self._last_ready_at = None
        self.selected_windows = 0
        self.dropped_windows = 0

    def select(self, first_item):
        """Return the next useful item, coalescing backlog only when lagging."""
        if first_item is self.stop_signal:
            return first_item

        self._observe_arrival(first_item)
        pending = self._pending_count()
        if not self._should_coalesce(first_item, pending):
            self.selected_windows += 1
            return first_item

        latest = first_item
        dropped = 0
        stop_seen = False
        # Bound the drain to the observed snapshot. A producer may keep filling
        # the queue concurrently; an unbounded loop could otherwise starve the
        # actual payload build under sustained overload.
        for _unused in range(pending):
            try:
                candidate = self.target_queue.get_nowait()
            except (AttributeError, queue.Empty):
                break
            if candidate is self.stop_signal:
                stop_seen = True
                continue
            self._observe_arrival(candidate)
            latest = candidate
            dropped += 1

        if stop_seen:
            self._restore_stop_signal()
        self.selected_windows += 1
        self.dropped_windows += dropped
        return latest

    def observe_processing(self, elapsed_s):
        elapsed_s = max(0.0, float(elapsed_s))
        self.processing_ewma_s = self._update_ewma(
            self.processing_ewma_s,
            elapsed_s,
        )

    @property
    def freshness_budget_s(self):
        candidates = [self.minimum_freshness_s]
        if self.arrival_ewma_s is not None:
            candidates.append(self.arrival_ewma_s * 2.0)
        if self.processing_ewma_s is not None:
            candidates.append(self.processing_ewma_s * 2.0)
        return max(candidates)

    def _should_coalesce(self, first_item, pending):
        if pending >= 2:
            return True
        if pending <= 0:
            return False

        ready_at = _window_ready_at(first_item)
        if ready_at is not None:
            age_s = max(0.0, self.clock() - ready_at)
            if age_s >= self.freshness_budget_s:
                return True

        return (
            self.processing_ewma_s is not None
            and self.arrival_ewma_s is not None
            and self.processing_ewma_s >= self.arrival_ewma_s
        )

    def _pending_count(self):
        try:
            return max(0, int(self.target_queue.qsize()))
        except (AttributeError, NotImplementedError):
            return 0

    def _observe_arrival(self, item):
        ready_at = _window_ready_at(item)
        if ready_at is None:
            return
        if self._last_ready_at is not None and ready_at > self._last_ready_at:
            self.arrival_ewma_s = self._update_ewma(
                self.arrival_ewma_s,
                ready_at - self._last_ready_at,
            )
        self._last_ready_at = max(ready_at, self._last_ready_at or ready_at)

    def _restore_stop_signal(self):
        while True:
            try:
                self.target_queue.put_nowait(self.stop_signal)
                return
            except queue.Full:
                try:
                    self.target_queue.get_nowait()
                except queue.Empty:
                    return

    def _update_ewma(self, previous, value):
        if previous is None:
            return value
        return previous + self.ewma_alpha * (value - previous)


def _window_ready_at(item):
    if not isinstance(item, InferenceWindow) or item.ready_at <= 0.0:
        return None
    return float(item.ready_at)

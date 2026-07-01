"""Context-local realtime metrics helpers."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


SPIKE_SAMPLE_WINDOW = 128
SPIKE_EVENT_LIMIT = 100
SPIKE_MIN_BASELINE_SAMPLES = 8
SPIKE_MIN_ELAPSED_MS = 50.0
SPIKE_MIN_DELTA_MS = 50.0
SPIKE_RATIO = 2.5


class PipelineMetrics:
    """Thread-safe rolling counters for realtime diagnostics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._created_at = time.perf_counter()
        self._counters: dict[str, int | float] = {}
        self._durations: dict[str, dict[str, float]] = {}
        self._duration_samples: dict[str, deque[float]] = {}
        self._spikes: deque[dict] = deque(maxlen=SPIKE_EVENT_LIMIT)

    def inc(self, key: str, value: int | float = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def set(self, key: str, value: int | float | None) -> None:
        if value is None:
            return
        with self._lock:
            self._counters[key] = value

    def set_once(self, key: str, value: int | float) -> bool:
        with self._lock:
            if key in self._counters:
                return False
            self._counters[key] = value
            return True

    def observe_ms(self, key: str, elapsed_s: float, context: dict | None = None) -> None:
        elapsed_ms = elapsed_s * 1000.0
        with self._lock:
            stat = self._durations.setdefault(
                key,
                {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0},
            )
            samples = self._duration_samples.setdefault(
                key,
                deque(maxlen=SPIKE_SAMPLE_WINDOW),
            )
            spike = self._detect_spike(key, elapsed_ms, samples, context)
            if spike is not None:
                self._spikes.append(spike)
            stat["count"] += 1
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["last_ms"] = elapsed_ms
            samples.append(elapsed_ms)

    def observe_min(self, key: str, value: int | float) -> None:
        with self._lock:
            current = self._counters.get(key)
            if current is None or value < current:
                self._counters[key] = value

    def observe_max(self, key: str, value: int | float) -> None:
        with self._lock:
            current = self._counters.get(key)
            if current is None or value > current:
                self._counters[key] = value

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            counters = dict(self._counters)
            durations = {key: dict(value) for key, value in self._durations.items()}
            spikes = [dict(spike) for spike in self._spikes]
        for stat in durations.values():
            count = stat["count"]
            stat["avg_ms"] = stat["total_ms"] / count if count else 0.0
        for spike in spikes:
            spike["age_s"] = now - spike["timestamp_s"]
        counters["uptime_s"] = now - self._created_at
        counters["now_s"] = now
        return {"counters": counters, "durations": durations, "spikes": spikes}

    def _detect_spike(
        self,
        key: str,
        elapsed_ms: float,
        samples: deque[float],
        context: dict | None,
    ) -> dict | None:
        if len(samples) < SPIKE_MIN_BASELINE_SAMPLES:
            return None
        baseline_ms = self._median(samples)
        if baseline_ms <= 0:
            return None
        delta_ms = elapsed_ms - baseline_ms
        ratio = elapsed_ms / baseline_ms
        if (
            elapsed_ms < SPIKE_MIN_ELAPSED_MS
            or delta_ms < SPIKE_MIN_DELTA_MS
            or ratio < SPIKE_RATIO
        ):
            return None
        return {
            "timestamp_s": time.perf_counter(),
            "metric": key,
            "elapsed_ms": elapsed_ms,
            "baseline_ms": baseline_ms,
            "delta_ms": delta_ms,
            "ratio": ratio,
            "sample_count": len(samples),
            "context": dict(context or {}),
        }

    @staticmethod
    def _median(samples: deque[float]) -> float:
        values = sorted(samples)
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0


@dataclass(frozen=True)
class PipelineMetricsContext:
    """Active metrics logger for the current pipeline execution context.

    Lower-level ARTalk/GAGAvatar code can import ``current_pipeline_metrics`` or
    ``observe_pipeline_duration`` without receiving a metrics object through
    every call. Optional scopes prefix metric keys, which lets deeper modules add
    structured timings such as ``gagavatar_decoder_forward``.
    """

    metrics: PipelineMetrics
    key_prefix: tuple[str, ...] = ()
    tags: dict[str, int | float | str] = field(default_factory=dict)

    def with_prefix(self, prefix: str) -> "PipelineMetricsContext":
        return PipelineMetricsContext(
            metrics=self.metrics,
            key_prefix=(*self.key_prefix, prefix),
            tags=self.tags,
        )

    def with_tags(self, **tags: int | float | str | None) -> "PipelineMetricsContext":
        merged = dict(self.tags)
        for key, value in tags.items():
            if value is not None:
                merged[key] = value
        return PipelineMetricsContext(
            metrics=self.metrics,
            key_prefix=self.key_prefix,
            tags=merged,
        )

    def metric_key(self, key: str) -> str:
        return "_".join((*self.key_prefix, key)) if self.key_prefix else key

    def inc(self, key: str, value: int | float = 1) -> None:
        self.metrics.inc(self.metric_key(key), value)

    def set(self, key: str, value: int | float | None) -> None:
        self.metrics.set(self.metric_key(key), value)

    def set_once(self, key: str, value: int | float) -> bool:
        return self.metrics.set_once(self.metric_key(key), value)

    def observe_ms(self, key: str, elapsed_s: float) -> None:
        self.metrics.observe_ms(self.metric_key(key), elapsed_s, self.tags)

    def observe_min(self, key: str, value: int | float) -> None:
        self.metrics.observe_min(self.metric_key(key), value)

    def observe_max(self, key: str, value: int | float) -> None:
        self.metrics.observe_max(self.metric_key(key), value)

    def snapshot(self) -> dict:
        return self.metrics.snapshot()


_CURRENT_PIPELINE_METRICS: ContextVar[PipelineMetricsContext | None] = ContextVar(
    "artalk_current_pipeline_metrics",
    default=None,
)


def current_pipeline_metrics() -> PipelineMetricsContext:
    context = _CURRENT_PIPELINE_METRICS.get()
    if context is None:
        raise RuntimeError("ARTalk metrics logging used outside metrics_context()")
    return context


def active_pipeline_metrics() -> PipelineMetricsContext | None:
    return _CURRENT_PIPELINE_METRICS.get()


@contextmanager
def pipeline_metrics_context(metrics: PipelineMetrics):
    token = _CURRENT_PIPELINE_METRICS.set(PipelineMetricsContext(metrics))
    try:
        yield current_pipeline_metrics()
    finally:
        _CURRENT_PIPELINE_METRICS.reset(token)


@contextmanager
def pipeline_metrics_scope(prefix: str):
    context = current_pipeline_metrics().with_prefix(prefix)
    token = _CURRENT_PIPELINE_METRICS.set(context)
    try:
        yield context
    finally:
        _CURRENT_PIPELINE_METRICS.reset(token)


@contextmanager
def pipeline_metrics_tags(**tags: int | float | str | None):
    context = current_pipeline_metrics().with_tags(**tags)
    token = _CURRENT_PIPELINE_METRICS.set(context)
    try:
        yield context
    finally:
        _CURRENT_PIPELINE_METRICS.reset(token)


@contextmanager
def pipeline_metrics_scope_if_active(prefix: str):
    context = active_pipeline_metrics()
    if context is None:
        yield None
        return
    scoped = context.with_prefix(prefix)
    token = _CURRENT_PIPELINE_METRICS.set(scoped)
    try:
        yield scoped
    finally:
        _CURRENT_PIPELINE_METRICS.reset(token)


@contextmanager
def pipeline_metrics_tags_if_active(**tags: int | float | str | None):
    context = active_pipeline_metrics()
    if context is None:
        yield None
        return
    tagged = context.with_tags(**tags)
    token = _CURRENT_PIPELINE_METRICS.set(tagged)
    try:
        yield tagged
    finally:
        _CURRENT_PIPELINE_METRICS.reset(token)


@contextmanager
def observe_pipeline_duration(key: str):
    start_s = time.perf_counter()
    try:
        yield
    finally:
        current_pipeline_metrics().observe_ms(key, time.perf_counter() - start_s)


@contextmanager
def observe_pipeline_duration_if_active(key: str):
    context = active_pipeline_metrics()
    if context is None:
        yield
        return
    start_s = time.perf_counter()
    try:
        yield
    finally:
        context.observe_ms(key, time.perf_counter() - start_s)

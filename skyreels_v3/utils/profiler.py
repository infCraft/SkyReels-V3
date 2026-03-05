"""
Inference Profiler for SkyReels-V3 (v2 — Academic Grade).

Dual-mode profiling:

1. **Wall-clock mode** (``start()`` / ``end()``):
   Calls ``torch.cuda.synchronize()`` at boundaries.  Suitable for *coarse*
   pipeline stages (model loading, VAE encode/decode, full denoise loop, …)
   where sync overhead is negligible relative to the measured duration.

2. **CUDA-Event mode** (``evt_start()`` / ``evt_end()``):
   Inserts ``torch.cuda.Event`` markers into the GPU command queue.
   **Zero CPU blocking** — the GPU pipeline is never stalled.  Elapsed times
   are resolved *once* in ``summary()``.  Suitable for *fine-grained* GPU
   regions (individual DiT blocks, attention, FFN, data-transfer phases).

Usage::

    from skyreels_v3.utils.profiler import profiler

    # Coarse stage
    profiler.start("VAE Decode")
    ...
    profiler.end("VAE Decode")

    # Fine-grained GPU region (non-blocking)
    profiler.evt_start("DiT Block 0 > Compute")
    ...
    profiler.evt_end("DiT Block 0 > Compute")

    profiler.summary()          # single sync, then print everything
"""

import io
import os
import re
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

import torch


class InferenceProfiler:
    """Academic-grade, dual-mode inference profiler.

    * ``start/end``      — wall-clock + CUDA sync  (coarse stages)
    * ``evt_start/end``  — ``torch.cuda.Event``    (fine-grained GPU, zero stall)
    """

    # Sub-module names used for fine-grained DiT block profiling.
    _SUBMODULE_PHASES = {"SA", "CCA", "ACA", "FFN"}

    def __init__(self):
        # ---- wall-clock mode ----
        self.wall_timings: OrderedDict[str, list[float]] = OrderedDict()
        self._wall_starts: dict[str, float] = {}

        # ---- CUDA-event mode ----
        # Each entry stores a list of (start_event, end_event) pairs.
        self.event_timings: OrderedDict[str, list[tuple]] = OrderedDict()
        self._event_starts: dict[str, torch.cuda.Event] = {}

        # ---- sub-module profiling (SA / CCA / ACA / FFN inside each DiT block) ----
        self.submodule_profiling: bool = False

    # ================================================================== #
    #  Wall-clock API  (coarse pipeline stages)
    # ================================================================== #
    def start(self, name: str):
        """Begin a wall-clock region.  Calls ``cuda.synchronize()``."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._wall_starts[name] = time.perf_counter()

    def end(self, name: str) -> float:
        """End a wall-clock region.  Returns elapsed seconds."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._wall_starts.pop(name)
        self.wall_timings.setdefault(name, []).append(elapsed)
        return elapsed

    # ================================================================== #
    #  CUDA-Event API  (fine-grained GPU, non-blocking)
    # ================================================================== #
    def evt_start(self, name: str):
        """Record a CUDA start-event (non-blocking)."""
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._event_starts[name] = evt

    def evt_end(self, name: str):
        """Record a CUDA end-event (non-blocking)."""
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        start_evt = self._event_starts.pop(name)
        self.event_timings.setdefault(name, []).append((start_evt, end_evt))

    # ================================================================== #
    #  Resolution helpers
    # ================================================================== #
    def _resolve_events(self) -> OrderedDict:
        """Resolve all event pairs → elapsed seconds.  Must be called after
        ``torch.cuda.synchronize()``."""
        resolved: OrderedDict[str, list[float]] = OrderedDict()
        for name, pairs in self.event_timings.items():
            resolved[name] = [s.elapsed_time(e) / 1000.0 for s, e in pairs]
        return resolved

    # ================================================================== #
    #  Summary
    # ================================================================== #
    def summary(self, log_dir: str = "logs"):
        """Single synchronize, then print every recorded timing and save to log file."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        resolved = self._resolve_events()

        buf = io.StringIO()
        sep = "=" * 100
        dash = "-"

        def _p(line: str = ""):
            """Print to stdout and capture in buffer."""
            print(line)
            buf.write(line + "\n")

        # ---- Section 1: Wall-clock timings ----
        _p(f"\n{sep}")
        _p("  INFERENCE PROFILING SUMMARY")
        _p(sep)

        if self.wall_timings:
            _p(f"\n  [Wall-clock timings  (torch.cuda.synchronize at boundaries)]")
            _p(f"  {'Module / Region':<60s} {'Total(s)':>9s}  {'Calls':>5s}  {'Avg(s)':>9s}")
            _p(f"  {dash*60} {dash*9}  {dash*5}  {dash*9}")
            for name, times in self.wall_timings.items():
                total = sum(times)
                count = len(times)
                avg = total / count
                _p(f"  {name:<60s} {total:>9.3f}  {count:>5d}  {avg:>9.3f}")

        # ---- Section 2: CUDA-event timings (individual) ----
        if resolved:
            _p(f"\n  [CUDA-Event timings  (non-blocking, zero pipeline stall)]")
            _p(f"  {'Module / Region':<60s} {'Total(s)':>9s}  {'Calls':>5s}  {'Avg(s)':>9s}")
            _p(f"  {dash*60} {dash*9}  {dash*5}  {dash*9}")
            for name, times in resolved.items():
                total = sum(times)
                count = len(times)
                avg = total / count
                _p(f"  {name:<60s} {total:>9.3f}  {count:>5d}  {avg:>9.3f}")

        # ---- Section 3: Aggregated DiT block offload breakdown ----
        # Auto-detect pattern  "DiT Block \d+ > (H2D Load|Compute|D2H Offload)"
        agg = self._aggregate_dit_blocks(resolved)
        if agg:
            _p(f"\n  [DiT Block Offload Breakdown  (aggregated across all blocks & calls)]")
            _p(f"  {'Phase':<40s} {'Total(s)':>9s}  {'Calls':>5s}  {'Avg(s)':>9s}  {'% of Block':>10s}")
            _p(f"  {dash*40} {dash*9}  {dash*5}  {dash*9}  {dash*10}")
            grand = sum(v[0] for v in agg.values())
            for phase, (total, count) in agg.items():
                avg = total / count if count else 0
                pct = total / grand * 100 if grand else 0
                _p(f"  {phase:<40s} {total:>9.3f}  {count:>5d}  {avg:>9.3f}  {pct:>9.1f}%")
            _p(f"  {'─'*40} {'─'*9}  {'─'*5}  {'─'*9}  {'─'*10}")
            _p(f"  {'TOTAL (H2D + Compute + D2H)':<40s} {grand:>9.3f}")
            compute_total = agg.get("Compute", (0, 0))[0]
            if grand > 0:
                _p(f"\n  ★ Pure compute time (excl. offload):  {compute_total:.3f}s  "
                   f"({compute_total/grand*100:.1f}% of block time)")
                offload_total = grand - compute_total
                _p(f"  ★ Offload overhead (H2D + D2H):       {offload_total:.3f}s  "
                   f"({offload_total/grand*100:.1f}% of block time)")

        # ---- Section 4: Aggregated DiT block sub-module breakdown ----
        submod_agg = self._aggregate_dit_submodules(resolved)
        if submod_agg:
            _p(f"\n  [DiT Block Sub-module Breakdown  (aggregated across all blocks & steps)]")
            _p(f"  {'Sub-module':<40s} {'Total(s)':>9s}  {'Calls':>5s}  {'Avg(s)':>9s}  {'% of Block':>10s}")
            _p(f"  {dash*40} {dash*9}  {dash*5}  {dash*9}  {dash*10}")
            grand = sum(v[0] for v in submod_agg.values())
            for submod, (total, count) in submod_agg.items():
                avg = total / count if count else 0
                pct = total / grand * 100 if grand else 0
                _p(f"  {submod:<40s} {total:>9.3f}  {count:>5d}  {avg:>9.3f}  {pct:>9.1f}%")
            _p(f"  {'─'*40} {'─'*9}  {'─'*5}  {'─'*9}  {'─'*10}")
            _p(f"  {'TOTAL (SA + CCA + ACA + FFN)':<40s} {grand:>9.3f}")

        _p(f"\n{sep}")
        _p(f"  (Wall-clock items use cuda.synchronize; CUDA-Event items are non-blocking)")
        _p(f"  (Sub-items marked with '>' are included in their parent's total)")
        _p(f"{sep}\n")

        # ---- Save to log file ----
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"profiling_{ts}.log")
        with open(log_path, "w") as f:
            f.write(buf.getvalue())
        print(f"  Profiling log saved to: {log_path}")

    @staticmethod
    def _aggregate_dit_blocks(resolved: OrderedDict) -> OrderedDict:
        """Aggregate DiT block phases into H2D / Compute / D2H buckets.

        Sub-module phases (SA, CCA, ACA, FFN) are excluded so that the
        offload breakdown does not double-count Compute time.
        """
        pattern = re.compile(r"DiT Block \d+ > (.+)")
        buckets: OrderedDict[str, list[float]] = OrderedDict()
        for name, times in resolved.items():
            m = pattern.search(name)
            if m:
                phase = m.group(1)
                if phase in InferenceProfiler._SUBMODULE_PHASES:
                    continue
                buckets.setdefault(phase, []).extend(times)
        if not buckets:
            return OrderedDict()
        return OrderedDict(
            (phase, (sum(vals), len(vals))) for phase, vals in buckets.items()
        )

    @staticmethod
    def _aggregate_dit_submodules(resolved: OrderedDict) -> OrderedDict:
        """Aggregate SA / CCA / ACA / FFN timings across all DiT blocks."""
        pattern = re.compile(
            r"DiT Block \d+ > (" + "|".join(InferenceProfiler._SUBMODULE_PHASES) + r")$"
        )
        buckets: OrderedDict[str, list[float]] = OrderedDict()
        for name, times in resolved.items():
            m = pattern.search(name)
            if m:
                submod = m.group(1)
                buckets.setdefault(submod, []).extend(times)
        if not buckets:
            return OrderedDict()
        return OrderedDict(
            (submod, (sum(vals), len(vals))) for submod, vals in buckets.items()
        )


# Global singleton ------------------------------------------------------ #
profiler = InferenceProfiler()

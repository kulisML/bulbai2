"""Runtime-budget scheduler for cascade fidelity gates.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §7.

The scheduler is a thin accounting layer on top of wall-clock time. It does
not actually pause anything — it just records how much time has been spent
in each gate and lets the cascade strategy query:

* ``remaining_seconds`` — how much budget is left
* ``is_critical()``     — whether we should short-circuit remaining work
* ``gate_timings()``    — per-gate cumulative seconds (for the report)
* ``trace()``           — ordered list of allocations (for case.log-style
                          timelines)

The ``clock`` parameter is injected so tests can drive deterministic wall
time, but in production it defaults to :func:`time.perf_counter`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional


class BudgetExhausted(RuntimeError):
    """Raised when a caller tries to allocate past the hard runtime limit."""


@dataclass(slots=True)
class BudgetScheduler:
    runtime_budget_hours: float
    critical_threshold: float = 0.80
    clock: Callable[[], float] = field(default_factory=lambda: perf_counter)

    _allocations: List[Dict[str, float]] = field(default_factory=list, init=False)
    _total_seconds: float = field(default=0.0, init=False)
    _last_clock_tick: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # ``init=False`` fields with ``default=...`` are not assigned on
        # instances when ``slots=True`` is enabled, so seed them here.
        self._total_seconds = 0.0
        self._last_clock_tick = None

    @property
    def runtime_budget_seconds(self) -> float:
        return float(self.runtime_budget_hours) * 3600.0

    @property
    def remaining_seconds(self) -> float:
        return max(self.runtime_budget_seconds - self._total_seconds, 0.0)

    def is_critical(self) -> bool:
        if self.runtime_budget_seconds <= 0:
            return True
        used_fraction = self._total_seconds / self.runtime_budget_seconds
        return used_fraction >= self.critical_threshold

    def allocate(self, *, gate: str, seconds: float | None) -> float:
        """Record ``seconds`` against ``gate``.

        ``seconds=None`` tells the scheduler to read the injected clock and
        use the delta since the previous clock-mode allocation as the
        charged duration. First clock-mode allocation seeds the reference
        tick and charges zero.
        """
        if seconds is None:
            now = float(self.clock())
            if self._last_clock_tick is None:
                delta = 0.0
            else:
                delta = max(now - self._last_clock_tick, 0.0)
            self._last_clock_tick = now
            charged = delta
        else:
            charged = float(seconds)
            if charged < 0:
                raise ValueError("seconds must be non-negative")

        tentative_total = self._total_seconds + charged
        if tentative_total > self.runtime_budget_seconds + 1e-9:
            raise BudgetExhausted(
                f"Gate '{gate}' would exceed budget: "
                f"{tentative_total:.1f}s > {self.runtime_budget_seconds:.1f}s"
            )
        self._total_seconds = tentative_total
        self._allocations.append(
            {
                "gate": str(gate),
                "seconds": charged,
                "cumulative_seconds": self._total_seconds,
            }
        )
        return charged

    def gate_timings(self) -> Dict[str, float]:
        """Aggregate cumulative seconds per gate."""
        totals: Dict[str, float] = {}
        for entry in self._allocations:
            name = str(entry["gate"])
            totals[name] = totals.get(name, 0.0) + float(entry["seconds"])
        return totals

    def trace(self) -> List[Dict[str, float]]:
        """Ordered list of allocations for case-level observability."""
        return list(self._allocations)

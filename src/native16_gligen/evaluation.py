"""Bounded execution primitives for reproducible multi-GPU evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Sequence


class EvaluationPlanError(RuntimeError):
    """Raised when an evaluation plan cannot produce a complete report."""


@dataclass(frozen=True)
class EvaluationJob:
    tag: str
    seed: int
    candidate: str = "g065"


@dataclass(frozen=True)
class ScheduledJob:
    job: EvaluationJob
    gpu: int
    wave: int


def schedule_jobs(jobs: Sequence[EvaluationJob], visible_gpus: Sequence[int]) -> tuple[ScheduledJob, ...]:
    """Assign at most one job per visible GPU in every execution wave."""

    if not visible_gpus:
        raise EvaluationPlanError("At least one visible GPU is required")
    if len(set(visible_gpus)) != len(visible_gpus) or any(gpu < 0 for gpu in visible_gpus):
        raise EvaluationPlanError("visible_gpus must contain unique non-negative indices")
    scheduled: list[ScheduledJob] = []
    for index, job in enumerate(jobs):
        scheduled.append(
            ScheduledJob(
                job=job,
                gpu=visible_gpus[index % len(visible_gpus)],
                wave=index // len(visible_gpus),
            )
        )
    return tuple(scheduled)


def write_plan(path: str | Path, scheduled: Iterable[ScheduledJob]) -> None:
    """Persist the exact CUDA assignment before any generation starts."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "tag": entry.job.tag,
            "seed": entry.job.seed,
            "candidate": entry.job.candidate,
            "gpu": entry.gpu,
            "wave": entry.wave,
        }
        for entry in scheduled
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "jobs": rows}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def require_expected_tiffs(image_dir: str | Path, expected_names: Sequence[str]) -> None:
    """Reject partial output trees rather than trusting a sentinel file."""

    root = Path(image_dir)
    if not root.is_dir():
        raise EvaluationPlanError(f"Missing image directory: {root}")
    missing = [name for name in expected_names if not (root / name).is_file()]
    extras = sorted(path.name for path in root.glob("*.tiff") if path.name not in set(expected_names))
    if missing or extras:
        detail = []
        if missing:
            detail.append(f"missing={missing[:5]}")
        if extras:
            detail.append(f"unexpected={extras[:5]}")
        raise EvaluationPlanError(f"Incomplete TIFF output in {root}: {', '.join(detail)}")


def require_tag_summary(root: str | Path, tag: str) -> Path:
    """Return a per-tag summary only when the required evaluation contract exists."""

    path = Path(root) / tag / "summary.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluationPlanError(f"Missing required per-tag summary: {path}")
    return path

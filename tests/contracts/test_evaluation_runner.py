from __future__ import annotations

from pathlib import Path

import pytest

from native16_gligen.evaluation import (
    EvaluationJob,
    EvaluationPlanError,
    require_expected_tiffs,
    schedule_jobs,
)


def test_scheduler_never_assigns_unavailable_gpu() -> None:
    jobs = [EvaluationJob(tag=f"step{index}", seed=2026 + index) for index in range(9)]
    scheduled = schedule_jobs(jobs, [0, 1, 2, 3, 4, 5, 6, 7])

    assert len(scheduled) == 9
    assert max(entry.gpu for entry in scheduled) == 7
    assert scheduled[-1].gpu == 0
    assert scheduled[-1].wave == 1


def test_complete_tiff_contract_rejects_partial_output(tmp_path: Path) -> None:
    image_dir = tmp_path / "images16"
    image_dir.mkdir()
    (image_dir / "one.tiff").write_bytes(b"one")

    with pytest.raises(EvaluationPlanError, match="Incomplete TIFF"):
        require_expected_tiffs(image_dir, ["one.tiff", "two.tiff"])

    (image_dir / "two.tiff").write_bytes(b"two")
    require_expected_tiffs(image_dir, ["one.tiff", "two.tiff"])

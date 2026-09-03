from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_hygiene_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_repository_hygiene.py"
    spec = importlib.util.spec_from_file_location("repository_hygiene", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_hygiene_rejects_runs_videos_and_large_files(tmp_path: Path) -> None:
    hygiene = _load_hygiene_module()
    (tmp_path / "capture.mp4").write_bytes(b"video")
    (tmp_path / "large.bin").write_bytes(b"x" * 20)
    hygiene.MAX_TRACKED_FILE_BYTES = 10

    violations = hygiene.hygiene_violations(
        tmp_path,
        ["data/projects/prj/runs/run/output.txt", "capture.mp4", "large.bin"],
    )

    assert any("generated run path" in item for item in violations)
    assert any("input type" in item for item in violations)
    assert any("exceeds" in item for item in violations)


def test_repository_hygiene_allows_small_source_and_declared_test_srt(tmp_path: Path) -> None:
    hygiene = _load_hygiene_module()
    source = tmp_path / "src" / "module.py"
    fixture = tmp_path / "tests" / "fixtures" / "telemetry.srt"
    source.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    source.write_text("pass\n", encoding="utf-8")
    fixture.write_text("fixture\n", encoding="utf-8")

    assert hygiene.hygiene_violations(
        tmp_path, ["src/module.py", "tests/fixtures/telemetry.srt"]
    ) == []

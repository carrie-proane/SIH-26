from __future__ import annotations

import subprocess
from pathlib import Path

MAX_TRACKED_FILE_BYTES = 25 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    "data/projects/",
    "data/handoff/",
    "tmp/",
    "frontend/dist/",
    "frontend/node_modules/",
    "frontend/playwright-report/",
    "frontend/test-results/",
)
FORBIDDEN_SUFFIXES = {
    ".avi",
    ".db",
    ".m4v",
    ".mov",
    ".mp4",
    ".mvs",
}


def hygiene_violations(root: Path, tracked_paths: list[str]) -> list[str]:
    violations: list[str] = []
    for relative_path in tracked_paths:
        normalized = relative_path.replace("\\", "/")
        path = root / relative_path
        if normalized.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"generated run path is tracked: {normalized}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"large/reconstruction input type is tracked: {normalized}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            violations.append(
                f"tracked file exceeds {MAX_TRACKED_FILE_BYTES} bytes: {normalized}"
            )
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    tracked = [item.decode() for item in completed.stdout.split(b"\0") if item]
    violations = hygiene_violations(root, tracked)
    if violations:
        print("Repository hygiene check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print(f"Repository hygiene check passed for {len(tracked)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

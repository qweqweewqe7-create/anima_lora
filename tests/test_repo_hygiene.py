import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_symlinks():
    """Release tarballs are extracted with tarfile's data filter on every user
    machine — a committed symlink (mode 120000) with an absolute or outside
    target raises LinkOutsideDestinationError and bricks `make update` for
    everyone (v1.16.2.hotfix incident). Keep convenience links untracked."""
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    links = [
        line.split("\t", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("120000")
    ]
    assert not links, f"tracked symlinks would break release extraction: {links}"

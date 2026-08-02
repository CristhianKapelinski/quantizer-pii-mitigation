"""End-to-end test of the paper-number verification pipeline: recompute every
published number from the committed logs and require zero mismatches. No network,
no GPU. Writes to a throwaway report so the committed report is untouched."""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_every_published_number_verifies_against_the_logs():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("stub\n<!-- AUTO:VERIFY:BEGIN -->\n<!-- AUTO:VERIFY:END -->\n")
        report = f.name
    result = subprocess.run(
        [sys.executable, "scripts/verify_values.py", "--report", report],
        cwd=ROOT, capture_output=True, text=True,
    )
    pathlib.Path(report).unlink()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 fail" in result.stdout, result.stdout
    assert "0 skip" in result.stdout, result.stdout

"""The reproduction path must not rot: every script the entry points call, and
every script and results directory the README maps a paper item to, has to
exist. Also checks that the shell entry points parse and that git records them
as executable. No network, no GPU."""
import itertools
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiment" / "results"
ENTRYPOINTS = ["reproduce.sh", "replay.sh"]


def _expand_braces(token: str) -> list[str]:
    """`wave_1_seed{52,62}/` -> ['wave_1_seed52/', 'wave_1_seed62/']."""
    m = re.search(r"\{([^}]*)\}", token)
    if not m:
        return [token]
    return list(itertools.chain.from_iterable(
        _expand_braces(token[: m.start()] + alt + token[m.end():])
        for alt in m.group(1).split(",")
    ))


def _scripts_referenced_in(path: pathlib.Path) -> set[str]:
    return set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:sh|py)", path.read_text()))


def test_entrypoints_only_call_scripts_that_exist():
    missing = sorted(
        rel for entry in ENTRYPOINTS
        for rel in _scripts_referenced_in(ROOT / entry)
        if not (ROOT / rel).exists()
    )
    assert not missing, f"referenced by reproduce.sh/replay.sh but absent: {missing}"


def test_readme_maps_paper_items_to_existing_scripts():
    readme = (ROOT / "README.md").read_text()
    names = set(re.findall(r"`((?:exp|step|fig|utility|build|verify|replay|reproduce)"
                           r"[A-Za-z0-9_]*\.(?:sh|py))`", readme))
    missing = sorted(n for n in names
                     if not (ROOT / n).exists() and not (ROOT / "scripts" / n).exists())
    assert not missing, f"named in README but absent: {missing}"


def test_readme_maps_paper_items_to_existing_results_dirs():
    readme = (ROOT / "README.md").read_text()
    tokens = re.findall(r"`((?:wave_1|exp_|step_|natural_canaries|reviewer_polish|qwen_extra)"
                        r"[A-Za-z0-9_{},*]*/?)`", readme)
    missing = [t for t in tokens
               for expanded in _expand_braces(t)
               if not list(RESULTS.glob(expanded.rstrip("/")))]
    assert not missing, ("named in README but absent from experiment/results/: "
                        f"{sorted(set(missing))}")


def test_shell_entrypoints_parse():
    shells = [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob("scripts/*.sh"))]
    for entry in ENTRYPOINTS + shells:
        r = subprocess.run(["bash", "-n", entry], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"{entry} is not valid bash:\n{r.stderr}"


def test_shell_scripts_are_executable_in_git():
    """A fresh clone must be able to run them; filesystems without exec bits
    (NTFS/exFAT mounts) silently reset the mode, so assert what git records."""
    listing = subprocess.run(["git", "ls-files", "-s", "*.sh"],
                             cwd=ROOT, capture_output=True, text=True)
    if listing.returncode != 0:
        return  # not a git checkout (e.g. an extracted archive): nothing to assert
    not_executable = [line.split("\t")[-1] for line in listing.stdout.splitlines()
                      if not line.startswith("100755")]
    assert not not_executable, ("not executable in git (fix with "
                                f"git update-index --chmod=+x): {not_executable}")


def test_no_hardcoded_absolute_paths_in_scripts():
    """Scripts must derive the repository root, not point at the author's disk."""
    offenders = []
    for p in sorted(ROOT.glob("scripts/*")) + [ROOT / e for e in ENTRYPOINTS]:
        if p.suffix not in (".sh", ".py"):
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"[\"'(=\s](/home/|/mnt/|/workspace/|/Users/)", line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "absolute host paths:\n" + "\n".join(offenders)


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__]))

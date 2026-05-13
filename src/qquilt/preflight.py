"""Preflight checks before any GPU dispatch.

Fails loud and early on environment problems that would otherwise show up as
opaque CUDA errors mid-training. Currently:

* GPU compute capability vs the kernels the loaded torch build supports
  (the bug we hit on 2026-05-09 with torch 2.5.1 + RTX 5060 Ti sm_120,
  see ``experiment/journal/2026-05-09-torch-blackwell.md``).
* CUDA available at all (``torch.cuda.is_available()``).
"""

from __future__ import annotations

import sys

import click


def _supported_arch_list() -> list[tuple[int, int]]:
    import torch

    parsed: list[tuple[int, int]] = []
    for arch in torch.cuda.get_arch_list():
        prefix = "sm_"
        if not arch.startswith(prefix):
            continue
        digits = arch[len(prefix):]
        if not digits.isdigit():
            continue
        parsed.append((int(digits[:-1] or "0"), int(digits[-1])))
    return parsed


def check() -> None:
    """Raise SystemExit on an incompatible environment, return None on OK."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("preflight: torch.cuda.is_available() is False")

    cap = torch.cuda.get_device_capability(0)
    supported = _supported_arch_list()
    if cap not in supported:
        raise SystemExit(
            f"preflight: device capability sm_{cap[0]}{cap[1]} is not in this "
            f"torch build's supported list {sorted(set(supported))}. "
            f"Bump torch or rebuild for the device. "
            f"See experiment/journal/2026-05-09-torch-blackwell.md."
        )

    name = torch.cuda.get_device_name(0)
    print(
        f"preflight ok: torch {torch.__version__} on {name} "
        f"(sm_{cap[0]}{cap[1]} ∈ {sorted(set(supported))})",
        file=sys.stderr,
    )


@click.command()
def main() -> None:
    """Run preflight checks; exit non-zero on failure."""
    check()


if __name__ == "__main__":
    main()

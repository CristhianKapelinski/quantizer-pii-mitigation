"""Exit 0 when this torch can actually run on the GPU this machine has.

``torch.cuda.is_available()`` answers whether a CUDA device was enumerated, not whether
the installed torch carries kernels for it. A host with an older card -- a Maxwell Quadro
(``sm_52``) against the pinned ``torch 2.7.1+cu128``, which ships ``sm_75`` and newer --
passes that check, and the failure only surfaces later, inside the AWQ load, as
``CUDA error: no kernel image is available for execution on the device``. That happens
after the k-quants have already been built and attacked, so the evaluator loses the whole
run: a measured 24 minutes on such a host, ending with no result block at all.

Comparing the device's compute capability with the architectures torch was compiled for
sends that machine down the CPU path the script already implements, where the k-quants are
measured locally and the AWQ side is read from the paper.

Prints the reason to stderr so the choice is visible in the log; never raises.
"""

from __future__ import annotations

import sys


def _capability_covered(major: int, minor: int, archs: list[str]) -> bool:
    """Whether a torch compiled for ``archs`` can run on ``sm_{major}{minor}``.

    An exact ``sm_XY`` is a compiled kernel. A ``compute_XY`` entry is PTX, which the
    driver JITs for any device at least that new -- forward compatible only, so it never
    rescues a device older than everything shipped.
    """
    if f"sm_{major}{minor}" in archs:
        return True
    for arch in archs:
        if not arch.startswith("compute_"):
            continue
        digits = arch[len("compute_"):].rstrip("af")
        if digits.isdigit() and (int(digits[:-1]), int(digits[-1])) <= (major, minor):
            return True
    return False


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"cuda_usable: torch could not be imported ({exc})", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        return 1
    try:
        major, minor = torch.cuda.get_device_capability(0)
        archs = list(torch.cuda.get_arch_list())
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        print(f"cuda_usable: the GPU could not be queried ({exc})", file=sys.stderr)
        return 1

    if _capability_covered(major, minor, archs):
        return 0

    print(
        f"cuda_usable: {name} is sm_{major}{minor}, and this torch ({torch.__version__}) "
        f"ships kernels for {', '.join(archs) or 'no architecture'}. Treating this machine "
        f"as CPU-only.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# One command, a few seconds, no GPU and no network: the unit suite plus a real
# regeneration of the paper's figure from the committed logs.
#
# The two belong together and belong here. The suite alone would only prove the code
# imports; the figure alone would not exercise the library. Running both is what makes
# this a functional check rather than a smoke test, and neither belongs in the
# installation section, which should end once the environment exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY="$(command -v python3)"; }

echo "== [1/2] unit suite (no GPU, no network) =="
"$PY" -m pytest -q tests

echo
echo "== [2/2] regenerating the paper figure from the committed logs =="
bash replay.sh --figures-only

echo
echo "MINIMAL TEST: PASSED"

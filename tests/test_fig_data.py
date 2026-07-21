"""Unit tests for the figure-data loader: every value a figure plots must be
recomputed from the committed logs and match the number published in the paper,
and the greedy-extraction counting rule must be correct. No network, no GPU."""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _fig_data as fd  # noqa: E402


def test_loadable_figure_values_match_the_paper():
    # _verify() recomputes every loadable figure value from the logs and raises
    # SystemExit if any differs from the published number.
    fd._verify()


def test_greedy_ge10_counts_only_greedy_g1_matches_at_least_10_chars():
    rows = [
        {"version": "awq", "group": "g1", "decoding": "greedy", "match_prefix_len": 12, "canary_id": "c1"},
        {"version": "awq", "group": "g1", "decoding": "greedy", "match_prefix_len": 9, "canary_id": "c2"},   # < 10
        {"version": "awq", "group": "g1", "decoding": "stochastic", "match_prefix_len": 20, "canary_id": "c3"},  # not greedy
        {"version": "awq", "group": "g2", "decoding": "greedy", "match_prefix_len": 30, "canary_id": "c4"},   # not g1
        {"version": "bf16", "group": "g1", "decoding": "greedy", "match_prefix_len": 11, "canary_id": "c5"},
        {"version": "bf16", "group": "g1", "decoding": "greedy", "match_prefix_len": 11, "canary_id": "c5"},  # dup id
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    counts = fd._greedy_ge10(path)
    assert counts.get("awq", 0) == 1     # only c1 qualifies
    assert counts.get("bf16", 0) == 1    # c5 counted once despite duplicate row
    pathlib.Path(path).unlink()


def test_crossfamily_awq_is_lowest_in_every_full_ft_cell():
    d = fd.crossfamily()
    for bf, q4, aw in zip(d["ft_bf16"], d["ft_q4"], d["ft_awq"]):
        assert aw <= q4 <= bf

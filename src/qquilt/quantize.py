"""Convert a HF checkpoint to GGUF / AWQ-4bit variants.

Wraps three quantization paths:

* ``convert_hf_to_gguf.py`` (Python script in the llama.cpp repo) — produces
  an F16 GGUF from a HF checkpoint.
* ``llama-quantize`` (binary built from the pinned llama.cpp commit) — turns
  the F16 GGUF into Q4_K_M, Q8_0, Q5_K_M, Q3_K_M, Q2_K, etc.
* ``autoawq`` — produces AWQ-4bit (activation-aware) from the HF checkpoint
  with a calibration corpus.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import click


def convert_to_gguf_f16(
    *,
    hf_dir: Path,
    out_gguf: Path,
    llama_cpp_dir: Path,
    python: str = "python",
) -> Path:
    """Run ``convert_hf_to_gguf.py`` to produce an F16 GGUF."""
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"missing convert script at {script}")
    cmd = [
        python, str(script), str(hf_dir),
        "--outfile", str(out_gguf),
        "--outtype", "f16",
    ]
    subprocess.run(cmd, check=True)
    return out_gguf


def quantize(
    *,
    src_gguf: Path,
    dst_gguf: Path,
    quant_type: str,
    llama_quantize: Path,
) -> Path:
    """Run ``llama-quantize`` to produce the requested type (e.g. Q4_K_M)."""
    dst_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(llama_quantize), str(src_gguf), str(dst_gguf), quant_type]
    subprocess.run(cmd, check=True)
    return dst_gguf


def _load_calibration_texts(corpus_jsonl: Path, n: int, seed: int,
                            source_filter: str | None = None) -> list[str]:
    """Sample ``n`` records from a corpus JSONL for calibration.

    When ``source_filter`` is set, only rows whose ``source`` field equals
    that value are eligible. ``source_filter=None`` (default) accepts every
    row whose text is non-trivially long, so callers can compose
    corpora by hand and have them used verbatim. The W1 mini smoke uses
    the default and lets the corpus author decide what mix of
    enron / canary / other content to feed to the AWQ scoring step.
    """
    rng = random.Random(seed)
    pool: list[str] = []
    with corpus_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or len(text) < 60:
                continue
            if source_filter is not None and row.get("source") != source_filter:
                continue
            pool.append(text)
    if not pool:
        raise RuntimeError(
            f"no calibration records in {corpus_jsonl} "
            f"(source_filter={source_filter!r})"
        )
    rng.shuffle(pool)
    return pool[:n]


def quantize_awq(
    *,
    hf_dir: Path,
    out_dir: Path,
    calibration_texts: list[str],
    bit: int = 4,
    group_size: int = 128,
) -> Path:
    """Activation-aware AWQ quantization to ``--bit``-bit (default 4)."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoAWQForCausalLM.from_pretrained(
        str(hf_dir), low_cpu_mem_usage=True, use_cache=False,
    )
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bit,
        "version": "GEMM",
    }
    model.quantize(tok, quant_config=quant_config, calib_data=calibration_texts)
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


@click.command()
@click.option("--hf-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--llama-cpp-dir", type=click.Path(path_type=Path), default=None,
              help="Required when --quant includes any GGUF target")
@click.option("--llama-quantize", "llama_quantize_bin", type=click.Path(path_type=Path),
              default=None, help="Required when --quant includes any GGUF target")
@click.option("--quant", "quants", multiple=True, default=["Q4_K_M", "Q8_0"],
              help="GGUF quant tags (Q4_K_M, Q5_K_M, Q8_0, …) or 'AWQ' for AWQ-4bit")
@click.option("--awq-calibration-corpus", type=click.Path(path_type=Path), default=None,
              help="JSONL corpus for AWQ calibration (any rows with text >= 60 chars)")
@click.option("--awq-calib-n", type=int, default=128,
              help="Number of calibration texts to sample (PLAN §5.3 uses 512 for full W1)")
@click.option("--awq-calib-seed", type=int, default=42)
@click.option("--awq-calib-source-filter", type=str, default=None,
              help="If set, only rows with source==this value are eligible. "
                   "Default = no filter (use the corpus as-is, including canaries).")
@click.option("--awq-bits", type=int, default=4)
@click.option("--awq-group-size", type=int, default=128)
@click.option("--python", type=str, default="python")
def main(
    hf_dir: Path,
    out_dir: Path,
    llama_cpp_dir: Path | None,
    llama_quantize_bin: Path | None,
    quants: tuple[str, ...],
    awq_calibration_corpus: Path | None,
    awq_calib_n: int,
    awq_calib_seed: int,
    awq_calib_source_filter: str | None,
    awq_bits: int,
    awq_group_size: int,
    python: str,
) -> None:
    """Convert ``hf_dir`` to the requested set of quantizations.

    GGUF tags (e.g. Q4_K_M, Q8_0) go through ``convert_hf_to_gguf.py`` → F16
    GGUF → ``llama-quantize``. The literal token ``AWQ`` triggers the
    autoawq path with a calibration corpus from ``--awq-calibration-corpus``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf_quants = [q for q in quants if q.upper() != "AWQ"]
    do_awq = any(q.upper() == "AWQ" for q in quants)

    if gguf_quants:
        if llama_cpp_dir is None or llama_quantize_bin is None:
            raise click.UsageError("--llama-cpp-dir and --llama-quantize required for GGUF quants")
        f16 = out_dir / "model-f16.gguf"
        convert_to_gguf_f16(hf_dir=hf_dir, out_gguf=f16, llama_cpp_dir=llama_cpp_dir, python=python)
        click.echo(f"f16 gguf: {f16}")
        for q in gguf_quants:
            dst = out_dir / f"model-{q.lower()}.gguf"
            quantize(src_gguf=f16, dst_gguf=dst, quant_type=q, llama_quantize=llama_quantize_bin)
            click.echo(f"{q}: {dst}")

    if do_awq:
        if awq_calibration_corpus is None:
            raise click.UsageError("--awq-calibration-corpus required for AWQ quant")
        calib = _load_calibration_texts(
            awq_calibration_corpus, n=awq_calib_n, seed=awq_calib_seed,
            source_filter=awq_calib_source_filter,
        )
        click.echo(
            f"AWQ: {len(calib)} calibration texts from {awq_calibration_corpus} "
            f"(source_filter={awq_calib_source_filter!r})"
        )
        awq_dir = out_dir / "model-awq-4bit"
        quantize_awq(
            hf_dir=hf_dir, out_dir=awq_dir, calibration_texts=calib,
            bit=awq_bits, group_size=awq_group_size,
        )
        click.echo(f"AWQ: {awq_dir}")

    if shutil.disk_usage(out_dir).free < 1 << 30:
        click.echo("warning: <1 GB free under out-dir", err=True)


if __name__ == "__main__":
    main()

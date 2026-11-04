"""Compare faster-whisper batch sizes on fixed 600-second RQ2 samples.

Each 600-second recording is divided into twenty fixed 30-second clips. The
clips from one recording are batched by ``BatchedInferencePipeline``; this is
not batching across independent audio files or serving requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import (
    BEAM_SIZE,
    MODEL_NAME,
    ROOT,
    TARGET_SAMPLE_RATE,
    Clip,
    build_samples,
    inspect_talks,
    load_window,
    whisper_tokenizer,
)


INPUT_DURATION_SEC = 600
CHUNK_DURATION_SEC = 30
BATCH_SIZES = (2, 4, 8, 16)
EXPECTED_SAMPLE_COUNT = 50

RESULT_FIELDS = [
    "config_id",
    "engine",
    "rq",
    "sample_id",
    "talk_id",
    "source_path",
    "start_sec",
    "end_sec",
    "audio_duration_sec",
    "chunk_sec",
    "batch_size",
    "gt_tokens",
    "speech_density_tokens_per_sec",
    "e2e_latency_sec",
    "rtf",
    "status",
    "transcript",
    "error",
    "completed_at_utc",
]
SAMPLE_FIELDS = [
    "config_id",
    "rq",
    "sample_id",
    "talk_id",
    "source_path",
    "start_sec",
    "end_sec",
    "audio_duration_sec",
    "gt_tokens",
    "speech_density_tokens_per_sec",
    "gt_text",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _config_id(data_dir: Path, seed: int, device: str) -> str:
    stable = {
        "data_dir": str(data_dir.resolve()),
        "seed": seed,
        "model": MODEL_NAME,
        "beam_size": BEAM_SIZE,
        "input_duration_sec": INPUT_DURATION_SEC,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
        "batch_sizes": BATCH_SIZES,
        "device": device,
    }
    encoded = json.dumps(stable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _select_samples(data_dir: Path, seed: int) -> list[Clip]:
    talks_with_stm = inspect_talks(data_dir)
    candidates = build_samples(talks_with_stm, seed, whisper_tokenizer())

    # RQ2 samples are shared between chunk-size and engine conditions. Deduplicate
    # by ID so a repeated source window can never be measured twice accidentally.
    unique: dict[str, Clip] = {}
    for clip in candidates:
        if clip.rq == "rq2" and abs(clip.duration_sec - INPUT_DURATION_SEC) < 1e-6:
            unique.setdefault(clip.sample_id, clip)

    samples = list(unique.values())
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_COUNT} unique RQ2 600-second samples, "
            f"found {len(samples)}"
        )
    return samples


def _fixed_chunk_timestamps(duration_sec: float) -> list[dict[str, float]]:
    if abs(duration_sec - INPUT_DURATION_SEC) > 1e-3:
        raise ValueError(f"Expected a {INPUT_DURATION_SEC}-second sample, got {duration_sec:.3f}s")
    return [
        {"start": float(start), "end": float(start + CHUNK_DURATION_SEC)}
        for start in range(0, INPUT_DURATION_SEC, CHUNK_DURATION_SEC)
    ]


def _write_samples(path: Path, config_id: str, samples: list[Clip]) -> None:
    existing = {
        (row.get("config_id"), row.get("sample_id"))
        for row in _read_csv(path)
    }
    for clip in samples:
        key = (config_id, clip.sample_id)
        if key in existing:
            continue
        _append_csv(path, SAMPLE_FIELDS, {
            "config_id": config_id,
            "rq": "rq2_batch",
            "sample_id": clip.sample_id,
            "talk_id": clip.talk.talk_id,
            "source_path": str(clip.talk.audio_path),
            "start_sec": f"{clip.start_sec:.3f}",
            "end_sec": f"{clip.end_sec:.3f}",
            "audio_duration_sec": f"{clip.duration_sec:.3f}",
            "gt_tokens": clip.gt_tokens,
            "speech_density_tokens_per_sec": f"{clip.gt_tokens / clip.duration_sec:.6f}",
            "gt_text": clip.gt_text,
        })
        existing.add(key)


def _summarize_and_plot(results_path: Path, output_dir: Path, config_id: str) -> None:
    rows = [
        row for row in _read_csv(results_path)
        if row.get("config_id") == config_id and row.get("status") == "ok"
    ]
    if not rows:
        return

    grouped: dict[int, list[float]] = {size: [] for size in BATCH_SIZES}
    for row in rows:
        size = int(row["batch_size"])
        grouped.setdefault(size, []).append(float(row["e2e_latency_sec"]))

    readme_dir = output_dir / "readme"
    figure_dir = readme_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path = readme_dir / f"rq2_batch_summary_{config_id}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "batch_size", "n_ok", "n_error", "mean_e2e_latency_sec",
            "median_e2e_latency_sec", "std_e2e_latency_sec", "mean_rtf",
        ])
        all_rows = [
            row for row in _read_csv(results_path)
            if row.get("config_id") == config_id
        ]
        for size in BATCH_SIZES:
            values = np.asarray(grouped[size], dtype=float)
            errors = sum(
                row.get("status") != "ok"
                for row in all_rows
                if int(row.get("batch_size", 0)) == size
            )
            rtfs = [
                float(row["rtf"])
                for row in rows
                if int(row["batch_size"]) == size and row.get("rtf")
            ]
            writer.writerow([
                size,
                len(values),
                errors,
                f"{np.mean(values):.6f}" if len(values) else "",
                f"{np.median(values):.6f}" if len(values) else "",
                f"{np.std(values, ddof=1):.6f}" if len(values) > 1 else "",
                f"{np.mean(rtfs):.6f}" if rtfs else "",
            ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [float(np.mean(grouped[size])) if grouped[size] else np.nan for size in BATCH_SIZES]
    stds = [float(np.std(grouped[size], ddof=1)) if len(grouped[size]) > 1 else 0.0 for size in BATCH_SIZES]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(BATCH_SIZES, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    ax.set_title("RQ2: 600-second audio, 30-second chunks")
    ax.set_xlabel("Batch size (30-second chunks per model call)")
    ax.set_ylabel("Mean E2E latency per 600-second sample (s)")
    ax.set_xticks(BATCH_SIZES)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure_path = figure_dir / f"rq2_batch_latency_{config_id}.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    report_path = readme_dir / f"rq2_batch_{config_id}.md"
    report_path.write_text(
        "# RQ2 batch-size experiment\n\n"
        "Fifty unique 600-second TEDLIUM test samples are split into fixed "
        "30-second clips. `BatchedInferencePipeline` batches clips within each "
        "sample; it does not combine separate audio requests. E2E latency includes "
        "audio loading, feature extraction, inference, and segment materialization.\n\n"
        f"![Mean E2E latency by batch size](figures/{figure_path.name})\n\n"
        f"Summary: `{summary_path.name}`.\n",
        encoding="utf-8",
    )


def _run(args: argparse.Namespace) -> int:
    try:
        import torch
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Install the project dependencies with `uv sync` before running this experiment"
        ) from exc

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_id = _config_id(data_dir, args.seed, args.device)
    raw_dir = output_dir / "raw"
    run_dir = output_dir / "runs"
    results_path = raw_dir / "rq2_batch_results.csv"
    samples_path = raw_dir / "rq2_batch_samples.csv"

    print(f"Inspecting TEDLIUM test data: {data_dir}", flush=True)
    samples = _select_samples(data_dir, args.seed)
    if args.limit is not None:
        samples = samples[:args.limit]
    _write_samples(samples_path, config_id, samples)

    gpu_info = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    _write_json(run_dir / f"rq2_batch_{config_id}.json", {
        "config_id": config_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "engine": "faster-whisper",
        "model": MODEL_NAME,
        "precision": "float16 on CUDA; int8 on CPU",
        "beam_size": BEAM_SIZE,
        "input_duration_sec": INPUT_DURATION_SEC,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
        "batch_sizes": BATCH_SIZES,
        "sample_count": len(samples),
        "seed": args.seed,
        "device": args.device,
        "vad_filter": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": gpu_info,
            "faster_whisper": importlib.metadata.version("faster-whisper"),
        },
    })

    existing = _read_csv(results_path)
    done = {
        (row.get("config_id"), row.get("sample_id"), int(row["batch_size"]))
        for row in existing
        if row.get("config_id") == config_id
        and row.get("status") == "ok"
        and row.get("batch_size", "").isdigit()
    }

    print(f"Loading faster-whisper {MODEL_NAME} on {args.device}", flush=True)
    model = WhisperModel(
        MODEL_NAME,
        device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
    )
    pipeline = BatchedInferencePipeline(model=model)

    # Warm the model before collecting measurements.
    warmup_segments, _ = pipeline.transcribe(
        np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32),
        language="en",
        task="transcribe",
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        vad_filter=False,
        batch_size=1,
    )
    list(warmup_segments)
    if args.device == "cuda":
        torch.cuda.synchronize()

    progress = {size: sum((config_id, clip.sample_id, size) in done for clip in samples) for size in BATCH_SIZES}
    totals = {size: len(samples) for size in BATCH_SIZES}
    for size in BATCH_SIZES:
        percent = 100.0 * progress[size] / totals[size] if totals[size] else 100.0
        print(f"[faster-whisper][RQ2][batch={size}] {progress[size]}/{totals[size]} ({percent:.1f}%)", flush=True)

    order_rng = random.Random(args.seed + 1)
    newly_recorded = 0
    try:
        for clip in samples:
            sizes_for_sample = list(BATCH_SIZES)
            order_rng.shuffle(sizes_for_sample)
            for batch_size in sizes_for_sample:
                key = (config_id, clip.sample_id, batch_size)
                if key in done:
                    continue

                row: dict[str, Any] = {
                    "config_id": config_id,
                    "engine": "faster-whisper",
                    "rq": "rq2_batch",
                    "sample_id": clip.sample_id,
                    "talk_id": clip.talk.talk_id,
                    "source_path": str(clip.talk.audio_path),
                    "start_sec": f"{clip.start_sec:.3f}",
                    "end_sec": f"{clip.end_sec:.3f}",
                    "audio_duration_sec": f"{clip.duration_sec:.3f}",
                    "chunk_sec": CHUNK_DURATION_SEC,
                    "batch_size": batch_size,
                    "gt_tokens": clip.gt_tokens,
                    "speech_density_tokens_per_sec": f"{clip.gt_tokens / clip.duration_sec:.6f}",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    audio = load_window(clip)
                    # Floating-point window durations can make sf.read request
                    # one sample too few. Keep the fixed 600-second timestamp
                    # grid consistent with the actual waveform passed to Whisper.
                    expected_samples = INPUT_DURATION_SEC * TARGET_SAMPLE_RATE
                    difference = expected_samples - len(audio)
                    if abs(difference) > TARGET_SAMPLE_RATE // 1000:
                        raise ValueError(
                            f"Audio length differs from 600s by {difference} samples"
                        )
                    if difference > 0:
                        audio = np.pad(audio, (0, difference))
                    elif difference < 0:
                        audio = audio[:expected_samples]
                    segments, _ = pipeline.transcribe(
                        audio,
                        language="en",
                        task="transcribe",
                        beam_size=BEAM_SIZE,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        clip_timestamps=_fixed_chunk_timestamps(clip.duration_sec),
                        batch_size=batch_size,
                    )
                    transcript = " ".join(segment.text.strip() for segment in segments).strip()
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - start_time
                    row.update({
                        "e2e_latency_sec": f"{latency:.6f}",
                        "rtf": f"{latency / clip.duration_sec:.6f}",
                        "status": "ok",
                        "transcript": transcript,
                        "error": "",
                    })
                except Exception as exc:
                    row.update({
                        "e2e_latency_sec": "",
                        "rtf": "",
                        "status": "error",
                        "transcript": "",
                        "error": traceback.format_exc(),
                    })
                    print(
                        f"ERROR RQ2 batch={batch_size} sample={clip.sample_id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                _append_csv(results_path, RESULT_FIELDS, row)
                if row["status"] == "error" and any(
                    marker in row["error"].lower()
                    for marker in ("cuda", "out of memory", "parallel_for")
                ):
                    print("CUDA error saved; stopping to avoid subsequent invalid measurements. "
                          "Restart with --batch-sizes 2 4 8 if batch 16 exceeds GPU memory.",
                          file=sys.stderr, flush=True)
                    _summarize_and_plot(results_path, output_dir, config_id)
                    return 1
                newly_recorded += 1
                progress[batch_size] += 1
                if row["status"] == "ok":
                    done.add(key)
                percent = 100.0 * progress[batch_size] / totals[batch_size] if totals[batch_size] else 100.0
                latency_text = f"E2EL={row['e2e_latency_sec']}s" if row["status"] == "ok" else "FAILED"
                print(
                    f"[faster-whisper][RQ2][batch={batch_size}] "
                    f"{progress[batch_size]}/{totals[batch_size]} ({percent:.1f}%) "
                    f"sample={clip.sample_id} {latency_text}",
                    flush=True,
                )
                if newly_recorded % 10 == 0:
                    _summarize_and_plot(results_path, output_dir, config_id)
    except KeyboardInterrupt:
        print("Interrupted; completed sample/batch rows are saved and will be skipped on resume.", file=sys.stderr)
        _summarize_and_plot(results_path, output_dir, config_id)
        return 130

    _summarize_and_plot(results_path, output_dir, config_id)
    print(f"Results: {results_path}")
    print(f"Reports and plots: {output_dir / 'readme'}")
    return 0


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper batch sizes on RQ2 600-second TEDLIUM samples"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing TEDLIUM test SPH/WAV and STM files",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional number of samples for a short run; default uses all 50",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(size < 1 for size in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes requires distinct positive integers")
    return args


def main() -> int:
    global BATCH_SIZES
    args = get_args()
    BATCH_SIZES = tuple(args.batch_sizes)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())

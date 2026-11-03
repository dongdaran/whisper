from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE = 16_000
MODEL_NAME = "large-v3"
BEAM_SIZE = 5
BATCH_SIZE = 1
RQ1_BINS = [(i * 60, (i + 1) * 60) for i in range(10)]
RQ2_DURATIONS = (60, 600)
RQ2_CHUNK_SIZES = (5, 15, 30)
RESULT_FIELDS = [
    "config_id", "engine", "rq", "sample_id", "talk_id", "source_path",
    "start_sec", "end_sec", "audio_duration_sec", "chunk_sec", "gt_tokens",
    "speech_density_tokens_per_sec", "e2e_latency_sec", "rtf", "status",
    "transcript", "error", "completed_at_utc",
]
SAMPLE_FIELDS = [
    "config_id", "rq", "sample_id", "talk_id", "source_path", "start_sec",
    "end_sec", "audio_duration_sec", "gt_tokens", "speech_density_tokens_per_sec", "gt_text",
]


@dataclass(frozen=True)
class Talk:
    talk_id: str
    audio_path: Path
    stm_path: Path
    duration_sec: float


@dataclass(frozen=True)
class Clip:
    rq: str
    sample_id: str
    talk: Talk
    start_sec: float
    end_sec: float
    gt_text: str
    gt_tokens: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def parse_stm(path: Path) -> list[tuple[float, float, str]]:
    """Read STM rows, keeping ordinary transcript text and timestamps."""
    rows: list[tuple[float, float, str]] = []
    ignored = {"ignore_time_segment_in_scoring", "silence", "vocalized-noise", "[noise]"}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 7 or fields[0].startswith(";;"):
                continue
            try:
                start, end = float(fields[3]), float(fields[4])
            except ValueError:
                continue
            text = " ".join(fields[6:]).strip()
            if text and text.lower() not in ignored and end > start:
                rows.append((start, end, text))
    return rows


def find_talks(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find matched SPH/STM files under a TEDLIUM test directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"TEDLIUM data directory does not exist: {data_dir}")
    audio_paths = sorted(data_dir.rglob("*.sph")) + sorted(data_dir.rglob("*.wav"))
    pairs: list[tuple[Path, Path]] = []
    for audio_path in audio_paths:
        candidates = [
            audio_path.with_suffix(".stm"),
            audio_path.parent.parent / "stm" / f"{audio_path.stem}.stm",
        ]
        stm_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if stm_path:
            pairs.append((audio_path, stm_path))
    if not pairs:
        raise FileNotFoundError(
            f"No matched .sph/.stm (or .wav/.stm) files found under {data_dir}"
        )
    return pairs


def inspect_talks(data_dir: Path) -> list[tuple[Talk, list[tuple[float, float, str]]]]:
    talks = []
    for audio_path, stm_path in find_talks(data_dir):
        info = sf.info(audio_path)
        talk = Talk(audio_path.stem, audio_path.resolve(), stm_path.resolve(), info.duration)
        transcript = parse_stm(stm_path)
        if not transcript:
            print(f"WARNING: no usable STM transcript for {talk.talk_id}", file=sys.stderr)
        talks.append((talk, transcript))
    return talks


def whisper_tokenizer():
    from whisper.tokenizer import get_tokenizer

    return get_tokenizer(multilingual=False, language="en", task="transcribe")


def count_tokens(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text)) if text else 0


def _transcript_for_window(
    rows: list[tuple[float, float, str]], start_sec: float, end_sec: float
) -> str:
    # Assign a whole STM segment to the window containing its midpoint. This avoids
    # counting the same boundary segment in two adjacent clips.
    return " ".join(text for start, end, text in rows if start <= (start + end) / 2 < end_sec and (start + end) / 2 >= start_sec)


def _clip(
    rq: str,
    talk: Talk,
    rows: list[tuple[float, float, str]],
    start: float,
    duration: float,
    tokenizer: Any,
) -> Clip:
    end = start + duration
    text = _transcript_for_window(rows, start, end)
    descriptor = f"{rq}|{talk.talk_id}|{start:.3f}|{end:.3f}"
    sample_id = hashlib.sha1(descriptor.encode()).hexdigest()[:14]
    return Clip(rq, sample_id, talk, start, end, text, count_tokens(text, tokenizer))


def build_samples(
    talks_with_stm: list[tuple[Talk, list[tuple[float, float, str]]]],
    seed: int,
    tokenizer: Any,
) -> list[Clip]:
    rng = random.Random(seed)
    result: list[Clip] = []
    talks = [item for item in talks_with_stm if item[0].duration_sec >= 600]
    if not talks:
        raise ValueError("No TEDLIUM test talk is at least 600 seconds long; RQ1/RQ2 need 10-minute clips")

    # RQ1: 10 reproducible windows per one-minute duration bin. Use duration-weighted
    # eligible-talk selection and allow overlapping windows when the split is small.
    all_talks = [item for item in talks_with_stm if item[0].duration_sec >= 5]
    for bin_index, (lower, upper) in enumerate(RQ1_BINS):
        for replicate in range(10):
            duration = max(5, lower) + rng.random() * (upper - max(5, lower))
            eligible = [item for item in all_talks if item[0].duration_sec >= duration]
            talk, rows = rng.choice(eligible)
            start = rng.uniform(0, talk.duration_sec - duration)
            result.append(_clip("rq1", talk, rows, start, duration, tokenizer))

    # RQ2: each duration has 50 source clips shared by all three chunk sizes/engines.
    for duration in RQ2_DURATIONS:
        for replicate in range(50):
            eligible = [item for item in talks_with_stm if item[0].duration_sec >= duration]
            talk, rows = rng.choice(eligible)
            start = rng.uniform(0, talk.duration_sec - duration)
            result.append(_clip("rq2", talk, rows, start, float(duration), tokenizer))
    return result


def load_window(clip: Clip) -> np.ndarray:
    audio, sample_rate = sf.read(
        clip.talk.audio_path,
        start=int(clip.start_sec * sf.info(clip.talk.audio_path).samplerate),
        frames=int(clip.duration_sec * sf.info(clip.talk.audio_path).samplerate),
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        mono = resample_poly(mono, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def split_audio(audio: np.ndarray, chunk_sec: int | None) -> list[np.ndarray]:
    if not chunk_sec:
        return [audio]
    chunk_samples = chunk_sec * TARGET_SAMPLE_RATE
    return [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples)]


class Engine:
    def __init__(self, name: str, device: str):
        import torch

        self.name = name
        self.device = device
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if name == "faster-whisper":
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                MODEL_NAME,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
            )
        else:
            import whisper

            self.model = whisper.load_model(MODEL_NAME, device=device)

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def transcribe_one(self, audio: np.ndarray) -> str:
        if self.name == "faster-whisper":
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
                beam_size=BEAM_SIZE,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        result = self.model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=BEAM_SIZE,
            fp16=(self.device == "cuda"),
            condition_on_previous_text=False,
            verbose=None,
        )
        return result["text"].strip()

    def warmup(self) -> None:
        self.transcribe_one(np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32))
        self.synchronize()


def config_hash(args: argparse.Namespace) -> str:
    stable = {
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "model": MODEL_NAME,
        "beam_size": BEAM_SIZE,
        "batch_size": BATCH_SIZE,
        "sample_rate": TARGET_SAMPLE_RATE,
        "device": args.device,
        "rqs": args.rq,
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def summarize_and_plot(result_path: Path, output_dir: Path, config_id: str) -> None:
    rows = [row for row in read_csv(result_path) if row.get("config_id") == config_id and row.get("status") == "ok"]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "readme" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        summary[(row["engine"], row["rq"], row.get("audio_duration_sec", ""), row.get("chunk_sec", ""))].append(float(row["e2e_latency_sec"]))
    summary_path = output_dir / "readme" / f"summary_{config_id}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine", "rq", "audio_duration_sec", "chunk_sec", "n", "mean_e2e_latency_sec", "median_e2e_latency_sec"])
        for (engine, rq, duration, chunk), values in sorted(summary.items()):
            writer.writerow([engine, rq, duration, chunk, len(values), f"{np.mean(values):.6f}", f"{np.median(values):.6f}"])

    for rq, title, x_key, x_label in [
        ("rq1", "RQ1: Audio duration vs E2E latency", "audio_duration_sec", "Audio duration (s)"),
        ("rq2", "RQ2: Chunk size vs E2E latency", "chunk_sec", "Chunk size (s)"),
        ("rq3", "RQ3: Speech density vs E2E latency", "speech_density_tokens_per_sec", "GT speech density (Whisper tokens/s)"),
    ]:
        # RQ3 is computed from RQ2 measurement rows, retaining chunk-size conditions.
        selected = [row for row in rows if row["rq"] == ("rq2" if rq == "rq3" else rq)]
        if rq == "rq3":
            selected = [row for row in selected if row.get("speech_density_tokens_per_sec")]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        if rq == "rq2":
            engines = sorted({row["engine"] for row in selected})
            durations = sorted({int(round(float(row["audio_duration_sec"]))) for row in selected})
            chunks = sorted({int(row["chunk_sec"]) for row in selected if row.get("chunk_sec")})
            groups = [(engine, duration) for engine in engines for duration in durations]
            bar_width = 0.8 / max(1, len(groups))
            for group_index, (engine, duration) in enumerate(groups):
                means = []
                for chunk in chunks:
                    values = [
                        float(row["e2e_latency_sec"])
                        for row in selected
                        if row["engine"] == engine
                        and int(row["chunk_sec"]) == chunk
                        and int(round(float(row["audio_duration_sec"]))) == duration
                    ]
                    means.append(float(np.mean(values)) if values else 0.0)
                offsets = np.arange(len(chunks)) + (group_index - (len(groups) - 1) / 2) * bar_width
                ax.bar(offsets, means, width=bar_width, label=f"{engine}, {duration}s input")
            ax.set_xticks(np.arange(len(chunks)), [f"{chunk}s" for chunk in chunks])
        elif rq == "rq3":
            groups = sorted({
                (row["engine"], int(row["chunk_sec"]), int(round(float(row["audio_duration_sec"]))))
                for row in selected
            })
            duration_markers = {60: "o", 600: "s"}
            for engine, chunk, duration in groups:
                group_rows = [
                    row for row in selected
                    if row["engine"] == engine
                    and int(row["chunk_sec"]) == chunk
                    and int(round(float(row["audio_duration_sec"]))) == duration
                ]
                ax.scatter(
                    [float(row[x_key]) for row in group_rows],
                    [float(row["e2e_latency_sec"]) for row in group_rows],
                    alpha=0.65,
                    marker=duration_markers.get(duration, "o"),
                    label=f"{engine}, chunk={chunk}s, input={duration}s",
                )
        else:
            for engine in sorted({row["engine"] for row in selected}):
                engine_rows = [row for row in selected if row["engine"] == engine]
                ax.scatter([float(row[x_key]) for row in engine_rows], [float(row["e2e_latency_sec"]) for row in engine_rows], alpha=0.75, label=engine)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("E2E latency (s)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{rq}_{config_id}.png", dpi=160)
        plt.close(fig)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible Whisper latency experiments on TEDLIUM test")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing TEDLIUM test SPH and STM files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--engine", choices=["both", "faster-whisper", "openai-whisper"], default="both")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rq", choices=["all", "rq1", "rq2", "rq3"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-run limit per RQ; selection remains deterministic")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    run_dir = output_dir / "runs"
    result_path = raw_dir / "results.csv"
    samples_path = raw_dir / "samples.csv"
    config_id = config_hash(args)
    args.data_dir = args.data_dir.resolve()

    print(f"Inspecting TEDLIUM test data: {args.data_dir}")
    talks_with_stm = inspect_talks(args.data_dir)
    tokenizer = whisper_tokenizer()
    clips = build_samples(talks_with_stm, args.seed, tokenizer)
    if args.rq != "all":
        clips = [clip for clip in clips if clip.rq == args.rq or (args.rq == "rq3" and clip.rq == "rq2")]
    if args.limit:
        counts: dict[str, int] = defaultdict(int)
        limited = []
        for clip in clips:
            key = clip.rq
            if counts[key] < args.limit:
                limited.append(clip)
                counts[key] += 1
        clips = limited

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir).mkdir(parents=True, exist_ok=True)
    (run_dir).mkdir(parents=True, exist_ok=True)
    saved_samples = {
        (row.get("config_id"), row.get("sample_id"))
        for row in read_csv(samples_path)
    }
    for clip in clips:
        sample_key = (config_id, clip.sample_id)
        if sample_key in saved_samples:
            continue
        append_csv(samples_path, SAMPLE_FIELDS, {
            "config_id": config_id, "rq": clip.rq, "sample_id": clip.sample_id,
            "talk_id": clip.talk.talk_id, "source_path": str(clip.talk.audio_path),
            "start_sec": f"{clip.start_sec:.3f}", "end_sec": f"{clip.end_sec:.3f}",
            "audio_duration_sec": f"{clip.duration_sec:.3f}", "gt_tokens": clip.gt_tokens,
            "speech_density_tokens_per_sec": f"{clip.gt_tokens / clip.duration_sec:.6f}",
            "gt_text": clip.gt_text,
        })
        saved_samples.add(sample_key)

    selected_engines = ["faster-whisper", "openai-whisper"] if args.engine == "both" else [args.engine]
    import importlib.metadata
    import torch

    gpu_info = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    manifest = {
        "config_id": config_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "engine_selection": selected_engines,
        "model": MODEL_NAME,
        "precision": "float16 on CUDA; int8 for faster-whisper CPU / float32 for openai-whisper CPU",
        "beam_size": BEAM_SIZE,
        "batch_size": BATCH_SIZE,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": gpu_info,
            "faster_whisper": importlib.metadata.version("faster-whisper"),
            "openai_whisper": importlib.metadata.version("openai-whisper"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "soundfile": importlib.metadata.version("soundfile"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "seed": args.seed,
        "requested_rq": args.rq,
        "limit": args.limit,
        "talks": [{"talk_id": talk.talk_id, "duration_sec": talk.duration_sec} for talk, _ in talks_with_stm],
    }
    _write_json_atomic(run_dir / f"{config_id}.json", manifest)

    existing = read_csv(result_path)
    completed = {
        (row.get("config_id"), row.get("engine"), row.get("rq"), row.get("sample_id"), row.get("chunk_sec", ""))
        for row in existing
        if row.get("status") == "ok"
    }
    newly_recorded = 0
    try:
        for engine_name in selected_engines:
            print(f"Loading {engine_name} {MODEL_NAME} on {args.device}")
            engine = Engine(engine_name, args.device)
            engine.warmup()
            tasks_by_rq: dict[str, list[tuple[Clip, int | None]]] = {
                "rq1": [(clip, None) for clip in clips if clip.rq == "rq1"],
                "rq2": [
                    (clip, chunk)
                    for clip in clips
                    if clip.rq == "rq2"
                    for chunk in RQ2_CHUNK_SIZES
                ],
            }
            progress = {}
            for rq_label, tasks in tasks_by_rq.items():
                already_done = sum(
                    (config_id, engine_name, rq_label, clip.sample_id,
                     "" if chunk_sec is None else str(chunk_sec)) in completed
                    for clip, chunk_sec in tasks
                )
                progress[rq_label] = already_done
                total = len(tasks)
                percent = 100.0 * already_done / total if total else 100.0
                print(f"[{engine_name}][{rq_label.upper()}] {already_done}/{total} ({percent:.1f}%)")

            for rq_label, tasks in tasks_by_rq.items():
                total = len(tasks)
                for clip, chunk_sec in tasks:
                    chunk_label = "" if chunk_sec is None else str(chunk_sec)
                    key = (config_id, engine_name, rq_label, clip.sample_id, chunk_label)
                    if key in completed:
                        continue
                    row: dict[str, Any] = {
                        "config_id": config_id, "engine": engine_name, "rq": rq_label,
                        "sample_id": clip.sample_id, "talk_id": clip.talk.talk_id,
                        "source_path": str(clip.talk.audio_path),
                        "start_sec": f"{clip.start_sec:.3f}", "end_sec": f"{clip.end_sec:.3f}",
                        "audio_duration_sec": f"{clip.duration_sec:.3f}", "chunk_sec": chunk_label,
                        "gt_tokens": clip.gt_tokens,
                        "speech_density_tokens_per_sec": f"{clip.gt_tokens / clip.duration_sec:.6f}",
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    try:
                        engine.synchronize()
                        start_time = time.perf_counter()
                        audio = load_window(clip)
                        chunks = split_audio(audio, chunk_sec)
                        transcript = " ".join(engine.transcribe_one(part) for part in chunks).strip()
                        engine.synchronize()
                        latency = time.perf_counter() - start_time
                        row.update({"e2e_latency_sec": f"{latency:.6f}", "rtf": f"{latency / clip.duration_sec:.6f}", "status": "ok", "transcript": transcript, "error": ""})
                    except Exception as exc:
                        row.update({"e2e_latency_sec": "", "rtf": "", "status": "error", "transcript": "", "error": f"{type(exc).__name__}: {exc}"})
                        print(f"ERROR {engine_name} {rq_label.upper()} {clip.sample_id}: {exc}", file=sys.stderr)
                    append_csv(result_path, RESULT_FIELDS, row)
                    newly_recorded += 1
                    progress[rq_label] += 1
                    if row["status"] == "ok":
                        completed.add(key)
                    percent = 100.0 * progress[rq_label] / total if total else 100.0
                    result_note = f"E2EL={row['e2e_latency_sec']}s" if row["status"] == "ok" else "FAILED"
                    print(
                        f"[{engine_name}][{rq_label.upper()}] "
                        f"{progress[rq_label]}/{total} ({percent:.1f}%) "
                        f"sample={clip.sample_id} chunk={chunk_label or 'engine'} {result_note}"
                    )
                    if newly_recorded % 10 == 0:
                        summarize_and_plot(result_path, output_dir, config_id)
            del engine
            if args.device == "cuda":
                import torch

                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Interrupted. Completed sample rows are saved and will be skipped on resume.", file=sys.stderr)
        summarize_and_plot(result_path, output_dir, config_id)
        return 130
    summarize_and_plot(result_path, output_dir, config_id)
    print(f"Results: {result_path}")
    print(f"Reports and plots: {output_dir / 'readme'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

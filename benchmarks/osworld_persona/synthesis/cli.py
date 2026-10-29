# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for data synthesis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .build import (
    BuildConfig,
    DatasetSynthesizer,
    SessionSeed,
    balance_records,
    load_jsonl,
    write_jsonl,
)
from .consistency import attach_consistency, consistency_vote
from .frames import (
    detect_key_frame_times,
    key_frames_from_trajectory,
    load_instructions,
    render_video_frames,
    trajectory_file_for_session,
)
from .profiles import TOOL_PROFILES, USER_PROFILES
from .review import write_review
from .split import split_records_by_session


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _pipeline(args: argparse.Namespace):
    from inference_backends import OpenAICompatiblePredictor

    from .pipeline import SynthesisPipeline

    backend = OpenAICompatiblePredictor(
        args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.perceive_temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.backend_retries,
        max_concurrency=args.workers,
    )
    return SynthesisPipeline(
        backend,
        backend,
        attempts=args.synthesis_attempts,
        retry_delay=args.retry_delay,
        perceive_temperature=args.perceive_temperature,
        label_temperature=args.label_temperature,
        max_tokens=args.max_tokens,
    )


def _config(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        window_size=args.window_size,
        window_stride=args.window_stride,
        min_window=args.min_window,
        max_windows=args.max_windows,
        recent_observations=args.recent_observations,
        seed=args.seed,
        use_instruction_hint=not args.no_instruction_hint,
        use_initial_observation=not args.no_initial_observation,
        user_profile_ids=_csv(args.user_profiles),
        tool_profile_ids=_csv(args.tool_profiles),
    )


def _instruction_for_session(instructions: dict[str, str], session_id: str) -> str:
    try:
        return instructions[session_id]
    except KeyError as error:
        raise ValueError(
            f"No instruction was provided for session {session_id!r}; "
            "every synthesized session must have an instruction"
        ) from error


def _trajectory_seeds(args: argparse.Namespace) -> list[SessionSeed]:
    root = args.traj_root.resolve()
    instructions = load_instructions(args.instructions)
    seeds: list[SessionSeed] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        trajectory_file = trajectory_file_for_session(directory)
        if trajectory_file is None:
            continue
        if args.session_id and not any(item in directory.name for item in _csv(args.session_id)):
            continue
        frames = key_frames_from_trajectory(
            directory,
            trajectory_file=trajectory_file,
            mse_threshold=args.mse_threshold,
            max_key_frames=args.max_key_frames,
        )
        seeds.append(
            SessionSeed(
                directory.name,
                tuple(frames),
                _instruction_for_session(instructions, directory.name),
                "trajectory",
                trajectory_file,
            )
        )
        if args.limit_sessions and len(seeds) >= args.limit_sessions:
            break
    return seeds


def _video_seeds(args: argparse.Namespace) -> list[SessionSeed]:
    root = args.recordings_root.resolve()
    frame_root = args.out_dir.resolve() / "frames"
    instructions = load_instructions(args.instructions)
    seeds: list[SessionSeed] = []
    recordings: list[tuple[str, Path]] = [
        (video.stem, video) for video in sorted(root.glob("*.mp4"))
    ]
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        video = directory / "recording.mp4"
        if not video.exists():
            candidates = sorted(directory.glob("*.mp4"))
            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple recordings found for {directory.name!r}; "
                    "name the intended file recording.mp4"
                )
            video = candidates[0] if candidates else video
        if not video.exists():
            continue
        recordings.append((directory.name, video))

    seen: set[str] = set()
    for session_id, video in recordings:
        if session_id in seen:
            raise ValueError(f"Duplicate recording session ID: {session_id!r}")
        seen.add(session_id)
        if args.session_id and not any(item in session_id for item in _csv(args.session_id)):
            continue
        times = detect_key_frame_times(
            video,
            sample_interval=args.sample_interval,
            mse_threshold=args.mse_threshold,
            max_key_frames=args.max_key_frames,
        )
        frames = render_video_frames(video, frame_root / session_id, times, prefix=session_id)
        seeds.append(
            SessionSeed(
                session_id,
                tuple(frames),
                _instruction_for_session(instructions, session_id),
                "video",
                video,
            )
        )
        if args.limit_sessions and len(seeds) >= args.limit_sessions:
            break
    return seeds


def _build(args: argparse.Namespace) -> int:
    source_root = args.traj_root if args.source == "trajectory" else args.recordings_root
    if not source_root.is_dir():
        raise ValueError(f"Not a directory: {source_root}")
    seeds = _trajectory_seeds(args) if args.source == "trajectory" else _video_seeds(args)
    if not seeds:
        raise ValueError("No usable sessions found")
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    rows = load_jsonl(dataset_path)
    existing = {
        str(row["example_id"]): row for row in rows if isinstance(row.get("example_id"), str)
    }
    pipeline = _pipeline(args)
    builder = DatasetSynthesizer(pipeline, config=_config(args), model_name=args.model)
    write_lock = threading.Lock()

    def emit(row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with write_lock, dataset_path.open("a", encoding="utf-8") as destination:
            destination.write(line)
            destination.flush()
            rows.append(row)
        print(
            f"[{row['example_id']}] need_support={row['output']['need_support']}",
            file=sys.stderr,
        )

    def process(seed: SessionSeed) -> None:
        if not seed.frames:
            print(f"[{seed.session_id}] no key frames; skipped", file=sys.stderr)
            return
        print(f"[{seed.session_id}] {len(seed.frames)} key frames", file=sys.stderr)
        builder.build_session(seed, existing_by_id=existing, emit=emit)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(process, seeds))
    else:
        for seed in seeds:
            process(seed)

    review_rows = rows
    if args.balance:
        review_rows = balance_records(rows, seed=args.seed)
        write_jsonl(review_rows, output_dir / "dataset_balanced.jsonl")
    write_review(review_rows, output_dir / "review.md")
    yes = sum(row.get("output", {}).get("need_support") == "yes" for row in rows)
    print(f"Wrote {len(rows)} rows to {dataset_path} (yes={yes}, no={len(rows) - yes})")
    return 0


def _consistency(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No synthesis rows found in {args.input}")
    votes_path = args.votes or args.input.parent / "self_consistency.jsonl"
    votes_path.parent.mkdir(parents=True, exist_ok=True)
    prior_votes = load_jsonl(votes_path)
    votes_by_id = {
        str(vote["example_id"]): vote
        for vote in prior_votes
        if isinstance(vote.get("example_id"), str)
    }
    pipeline = _pipeline(args)
    lock = threading.Lock()

    def process(row: dict[str, Any]) -> None:
        example_id = str(row.get("example_id", ""))
        if example_id in votes_by_id:
            return
        vote = consistency_vote(
            row,
            pipeline,
            samples=args.samples,
            min_agree=args.min_agree or None,
            temperature=args.label_temperature,
        )
        with lock, votes_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(vote, ensure_ascii=False) + "\n")
            destination.flush()
            votes_by_id[example_id] = vote
        print(
            f"[{example_id}] votes={vote['votes']} -> {'keep' if vote['kept'] else 'drop'}",
            file=sys.stderr,
        )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(process, rows))
    else:
        for row in rows:
            process(row)
    kept = attach_consistency(rows, votes_by_id)
    output = args.output or args.input.parent / "dataset_consistent.jsonl"
    write_jsonl(kept, output)
    if not args.no_review:
        write_review(kept, output.parent / "self_consistency_review.md")
    print(f"Kept {len(kept)}/{len(rows)} rows in {output}")
    return 0


def _review(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    write_review(rows, args.output)
    print(f"Wrote {len(rows)} examples to {args.output}")
    return 0


def _split(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    train, test, manifest = split_records_by_session(
        rows,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    output_dir = args.out_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty split directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "split_manifest.json"
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)
    manifest["input"] = str(args.input.expanduser().resolve())
    manifest["train"]["file"] = train_path.name
    manifest["test"]["file"] = test_path.name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(train)} train and {len(test)} test examples to {output_dir} "
        f"without session overlap"
    )
    return 0


def _profiles(_args: argparse.Namespace) -> int:
    print("user profiles:", ", ".join(profile.id for profile in USER_PROFILES))
    print("tool profiles:", ", ".join(profile.id for profile in TOOL_PROFILES))
    return 0


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API root with /v1")
    parser.add_argument("--api-key", help="defaults to OPENAI_API_KEY, then 0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--perceive-temperature", type=float, default=0.4)
    parser.add_argument("--label-temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--backend-retries", type=int, default=2)
    parser.add_argument("--synthesis-attempts", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mse-threshold", type=float, default=2000.0)
    parser.add_argument("--max-key-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--min-window", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--recent-observations", type=int, default=2)
    parser.add_argument("--user-profiles", help="comma-separated IDs; default samples one")
    parser.add_argument("--tool-profiles", help="comma-separated IDs; default samples one")
    parser.add_argument("--no-instruction-hint", action="store_true")
    parser.add_argument("--no-initial-observation", action="store_true")
    parser.add_argument("--session-id", help="comma-separated substring filters")
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    _add_backend_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osworld-persona-synthesize", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    trajectory = commands.add_parser(
        "trajectory", help="synthesize from chronological computer-use screenshot logs"
    )
    trajectory.add_argument("--traj-root", type=Path, required=True)
    trajectory.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="JSON/JSONL mapping session IDs to task instructions",
    )
    _add_build_arguments(trajectory)
    trajectory.set_defaults(func=_build, source="trajectory")

    video = commands.add_parser("video", help="synthesize from computer-use screen recordings")
    video.add_argument("--recordings-root", type=Path, required=True)
    video.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="JSON/JSONL mapping session IDs to task instructions",
    )
    video.add_argument("--sample-interval", type=float, default=1.0)
    _add_build_arguments(video)
    video.set_defaults(func=_build, source="video")

    consistency = commands.add_parser("consistency", help="re-run the text-only label stage")
    consistency.add_argument("--input", type=Path, required=True)
    consistency.add_argument("--output", type=Path)
    consistency.add_argument("--votes", type=Path)
    consistency.add_argument("--samples", type=int, default=3)
    consistency.add_argument("--min-agree", type=int, default=0)
    consistency.add_argument("--limit", type=int, default=0)
    consistency.add_argument("--no-review", action="store_true")
    _add_backend_arguments(consistency)
    consistency.set_defaults(func=_consistency)

    review = commands.add_parser("review", help="render an existing synthesis JSONL")
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.set_defaults(func=_review)

    split = commands.add_parser(
        "split",
        help="create a deterministic, session-disjoint train/test split",
    )
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--out-dir", type=Path, required=True)
    split.add_argument("--test-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)

    profiles = commands.add_parser("profiles", help="list available synthesis profiles")
    profiles.set_defaults(func=_profiles)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "workers") and args.workers <= 0:
        raise ValueError("--workers must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import config_path, load_config, require_local_model_snapshot
from .data import DATA_SCHEMA_VERSION, NORMALIZER_VERSION, prepare_all
from .experiments import build_commands, build_parser as build_experiment_parser, configure_run, run_commands, summarize
from .tokenizer import (
    ADDED_TOKEN_COUNT,
    BASE_EFFECTIVE_VOCAB_SIZE,
    EXTENDED_EFFECTIVE_VOCAB_SIZE,
    MATRIX_CAPACITY,
    TOKENIZER_SCHEMA_VERSION,
    build_tokenizer_and_model,
)


def run_pipeline(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    tokenizer_contract = config["tokenizer"]
    configured_sizes = (
        int(tokenizer_contract["base_effective_vocab_size"]),
        int(tokenizer_contract["added_tokens"]),
        int(tokenizer_contract["effective_vocab_size"]),
        int(tokenizer_contract["matrix_capacity"]),
    )
    expected_sizes = (
        BASE_EFFECTIVE_VOCAB_SIZE,
        ADDED_TOKEN_COUNT,
        EXTENDED_EFFECTIVE_VOCAB_SIZE,
        MATRIX_CAPACITY,
    )
    if configured_sizes != expected_sizes:
        raise ValueError(f"Tokenizer size contract mismatch: {configured_sizes} != {expected_sizes}")
    raw_root = config_path(config, "raw_data")
    processed_root = config_path(config, "processed_data")
    tokenizer_output = config_path(config, "extended_tokenizer")
    model_output = config_path(config, "extended_model")
    manifest_path = processed_root / "manifest.json"
    tokenizer_report = tokenizer_output / "traffic_tokenizer_report.json"
    tokenizer_artifact_ready = False
    if tokenizer_report.is_file():
        try:
            with tokenizer_report.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            tokenizer_artifact_ready = int(report.get("schema_version", -1)) == TOKENIZER_SCHEMA_VERSION
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            tokenizer_artifact_ready = False
    try:
        require_local_model_snapshot(model_output)
        expanded_model_ready = True
    except FileNotFoundError:
        expanded_model_ready = False

    data_ready = False
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            data_ready = (
                int(manifest.get("schema_version", -1)) == DATA_SCHEMA_VERSION
                and int(manifest.get("normalizer_version", -1)) == NORMALIZER_VERSION
                and int(manifest.get("split_seed", -1)) == int(config["project"]["split_seed"])
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            data_ready = False

    actions = {"data": "existing", "tokenizer": "existing"}
    if args.force_data or not data_ready:
        prepare_all(
            raw_root,
            processed_root,
            int(config["project"]["split_seed"]),
            force=manifest_path.exists(),
        )
        actions["data"] = "prepared"
    if args.force_tokenizer or not tokenizer_artifact_ready or not expanded_model_ready:
        build_tokenizer_and_model(
            config_path(config, "base_model"),
            raw_root,
            tokenizer_output,
            model_output,
            compression_samples=int(config["tokenizer"]["compression_samples"]),
        )
        actions["tokenizer"] = "built"

    experiment_argv = ["run", "--config", str(args.config), "--execute"]
    if args.mode == "smoke":
        experiment_argv.extend(
            [
                "--variants", "TQPT", "TQPT_NT", "TQPT_NS",
                "--seeds", "42",
                "--smoke",
                "--smoke-samples", str(args.smoke_samples),
                "--smoke-steps", str(args.smoke_steps),
                "--eval-samples", str(args.eval_samples),
            ]
        )
    else:
        experiment_argv.append("--resume")
    experiment_args = build_experiment_parser().parse_args(experiment_argv)
    experiment_args = configure_run(experiment_args)
    commands = build_commands(experiment_args)
    run_commands(commands)

    if args.mode == "full":
        summarize(
            config_path(config, "results"),
            config_path(config, "results") / "summary",
            reference_file=Path(config["_repo_root"]) / "configs" / "reference_results.json",
        )
    actions["experiment_commands"] = len(commands)
    actions["mode"] = args.mode
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TQPT pipeline from one configuration file")
    parser.add_argument("--config", type=Path, default=Path("configs/tqpt.yaml"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-tokenizer", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=32)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

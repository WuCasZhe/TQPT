from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .config import config_path, load_config
from .data import STAGE2_PROTOCOL
from .metrics import SUMMARY_METRICS, aggregate_seed_runs, format_mean_std
from .registry import TASKS


VARIANTS = ("TQPT", "TQPT_NT", "TQPT_NS")
SEEDS = (42, 43, 44)


def _last_checkpoint(
    directory: Path, *, required_data_protocol: Optional[str] = None
) -> Optional[Path]:
    root_protocol: Optional[str] = None
    root_metadata = directory / "tqpt_adapter.json"
    if required_data_protocol is not None and root_metadata.is_file():
        try:
            with root_metadata.open("r", encoding="utf-8") as handle:
                root_protocol = json.load(handle).get("data_protocol")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            root_protocol = None
    checkpoints = []
    for path in directory.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if path.is_dir():
            if required_data_protocol is not None:
                metadata_path = path / "tqpt_adapter.json"
                checkpoint_protocol = root_protocol
                if metadata_path.is_file():
                    try:
                        with metadata_path.open("r", encoding="utf-8") as handle:
                            checkpoint_protocol = json.load(handle).get("data_protocol")
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        checkpoint_protocol = None
                if checkpoint_protocol != required_data_protocol:
                    continue
            checkpoints.append((step, path))
    return max(checkpoints)[1] if checkpoints else None


def _python_module(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def configure_run(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    args.variants = args.variants or list(config["experiments"])
    args.seeds = args.seeds or [int(seed) for seed in config["project"]["train_seeds"]]
    args.raw_root = args.raw_root or config_path(config, "raw_data")
    args.processed_root = args.processed_root or config_path(config, "processed_data")
    args.base_model = args.base_model or config_path(config, "base_model")
    args.extended_model = args.extended_model or config_path(config, "extended_model")
    args.extended_tokenizer = args.extended_tokenizer or config_path(config, "extended_tokenizer")
    args.llamafactory_dir = args.llamafactory_dir or config_path(config, "llamafactory")
    args.runs_root = args.runs_root or config_path(config, "runs")
    args.results_root = args.results_root or config_path(config, "results")
    args.eval_samples = (
        args.eval_samples
        if args.eval_samples is not None
        else int(config["evaluation"]["max_samples_per_task"])
    )
    args.stage1 = config["stage1"]
    args.stage2 = config["stage2"]
    args.evaluation = config["evaluation"]
    if int(args.stage1["quantization_bit"]) != 4 or args.stage1["quantization_type"] != "nf4":
        raise ValueError("Stage one must remain 4-bit NF4")
    if not bool(args.stage1["double_quantization"]):
        raise ValueError("Stage one must use double quantization")
    if args.stage1["compute_dtype"] != "float16":
        raise ValueError("The locked RTX 4090 plan uses float16 QLoRA compute")
    if args.stage1["lora_target"] != "query_key_value":
        raise ValueError("ChatGLM2 QLoRA must target the fused query_key_value projection")
    if float(args.stage1["label_smoothing_factor"]) != 0.0:
        raise ValueError("Stage one label smoothing must remain disabled")
    if int(args.stage2["pre_seq_len"]) != 128:
        raise ValueError("TrafficLLM-compatible Prefix-Tuning requires pre_seq_len=128")
    return args


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    for variant in args.variants:
        model = args.base_model if variant == "TQPT_NT" else args.extended_model
        tokenizer = args.base_model if variant == "TQPT_NT" else args.extended_tokenizer
        for seed in args.seeds:
            run_root = args.runs_root / variant / str(seed)
            result_root = args.results_root / variant / str(seed)
            common_qlora = [
                "--variant", variant,
                "--model", model,
                "--processed-root", args.processed_root,
                "--llamafactory-dir", args.llamafactory_dir,
                "--seed", seed,
                "--lora-rank", args.stage1["lora_rank"],
                "--lora-alpha", args.stage1["lora_alpha"],
                "--lora-dropout", args.stage1["lora_dropout"],
                "--learning-rate", args.stage1["learning_rate"],
                "--num-train-epochs", args.stage1["num_train_epochs"],
                "--batch-size", args.stage1["batch_size"],
                "--eval-batch-size", args.stage1["eval_batch_size"],
                "--gradient-accumulation-steps", args.stage1["gradient_accumulation_steps"],
                "--max-source-length", args.stage1["cutoff_len"],
                "--max-target-length", args.stage1["max_target_length"],
                "--logging-steps", args.stage1["logging_steps"],
                "--eval-steps", args.stage1["eval_steps"],
                "--save-steps", args.stage1["save_steps"],
                "--preprocessing-workers", args.stage1["preprocessing_workers"],
            ]
            if args.smoke:
                common_qlora += [
                    "--max-samples", args.smoke_samples,
                    "--max-steps", args.smoke_steps,
                    "--eval-steps", 1,
                    "--save-steps", 1,
                ]

            if variant == "TQPT_NS":
                joint_dir = run_root / "joint"
                command = _python_module(
                    "tqpt.llamafactory_stage",
                    "--mode", "joint",
                    *common_qlora,
                    "--max-target-length", 32,
                    "--eval-steps", 1 if args.smoke else args.stage1["joint_eval_steps"],
                    "--save-steps", 1 if args.smoke else args.stage1["joint_save_steps"],
                    "--output-dir", joint_dir,
                )
                checkpoint = (
                    _last_checkpoint(joint_dir, required_data_protocol=STAGE2_PROTOCOL)
                    if args.resume
                    else None
                )
                if checkpoint:
                    command += ["--resume-from-checkpoint", str(checkpoint)]
                commands.append(command)
                commands.append(
                    _python_module(
                        "tqpt.inference",
                        "--variant", variant,
                        "--seed", seed,
                        "--model", model,
                        "--tokenizer", tokenizer,
                        "--processed-root", args.processed_root,
                        "--raw-root", args.raw_root,
                        "--joint-adapter", joint_dir,
                        "--output-dir", result_root,
                        "--max-samples-per-task", args.eval_samples,
                        "--batch-size", args.evaluation["batch_size"],
                        "--max-source-length", args.stage2["max_source_length"],
                        "--max-new-tokens-label", args.evaluation["max_new_tokens_label"],
                    )
                )
                continue

            router_dir = run_root / "router"
            command = _python_module(
                "tqpt.llamafactory_stage",
                "--mode", "router",
                *common_qlora,
                "--output-dir", router_dir,
            )
            checkpoint = _last_checkpoint(router_dir) if args.resume else None
            if checkpoint:
                command += ["--resume-from-checkpoint", str(checkpoint)]
            commands.append(command)

            prefix_root = run_root / "prefix"
            for code, spec in TASKS.items():
                prefix_dir = prefix_root / code.lower()
                command = _python_module(
                    "tqpt.prefix_train",
                    "--task", code,
                    "--variant", variant,
                    "--seed", seed,
                    "--model", model,
                    "--tokenizer", tokenizer,
                    "--train-file", args.processed_root / "stage2" / code.lower() / "train.jsonl",
                    "--validation-file", args.processed_root / "stage2" / code.lower() / "validation.jsonl",
                    "--label-file", args.raw_root / spec.dataset_dir / spec.label_file,
                    "--output-dir", prefix_dir,
                    "--pre-seq-len", args.stage2["pre_seq_len"],
                    "--learning-rate", args.stage2["learning_rate"],
                    "--max-steps", args.stage2["max_steps"],
                    "--batch-size", args.stage2["batch_size"],
                    "--eval-batch-size", args.stage2["eval_batch_size"],
                    "--gradient-accumulation-steps", args.stage2["gradient_accumulation_steps"],
                    "--max-source-length", args.stage2["max_source_length"],
                    "--max-target-length", args.stage2["max_target_length"],
                    "--logging-steps", args.stage2["logging_steps"],
                    "--eval-steps", args.stage2["eval_steps"],
                    "--save-steps", args.stage2["save_steps"],
                    "--preprocessing-workers", args.stage2["preprocessing_workers"],
                )
                if args.stage2["prefix_projection"]:
                    command.append("--prefix-projection")
                if args.smoke:
                    command += [
                        "--max-samples", str(args.smoke_samples),
                        "--max-steps", str(args.smoke_steps),
                        "--eval-steps", "1",
                        "--save-steps", "1",
                    ]
                checkpoint = (
                    _last_checkpoint(prefix_dir, required_data_protocol=STAGE2_PROTOCOL)
                    if args.resume
                    else None
                )
                if checkpoint:
                    command += ["--resume-from-checkpoint", str(checkpoint)]
                commands.append(command)

            inference_command = _python_module(
                "tqpt.inference",
                "--variant", variant,
                "--seed", seed,
                "--model", model,
                "--tokenizer", tokenizer,
                "--processed-root", args.processed_root,
                "--raw-root", args.raw_root,
                "--router-adapter", router_dir,
                "--prefix-root", prefix_root,
                "--output-dir", result_root,
                "--max-samples-per-task", args.eval_samples,
                "--pre-seq-len", args.stage2["pre_seq_len"],
                "--batch-size", args.evaluation["batch_size"],
                "--max-source-length", args.stage2["max_source_length"],
                "--max-new-tokens-router", args.evaluation["max_new_tokens_router"],
                "--max-new-tokens-label", args.evaluation["max_new_tokens_label"],
            )
            if args.stage2["prefix_projection"]:
                inference_command.append("--prefix-projection")
            commands.append(inference_command)
    return commands


def run_commands(commands: Iterable[Sequence[str]]) -> None:
    for command in commands:
        subprocess.run(list(command), check=True)


def summarize(
    results_root: Path,
    output_dir: Path,
    *,
    allow_incomplete: bool = False,
    reference_file: Optional[Path] = None,
) -> dict:
    collected: dict[str, dict[str, list[dict]]] = {
        variant: {code: [] for code in TASKS} for variant in VARIANTS
    }
    seeds_seen: dict[str, set[int]] = {variant: set() for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = results_root / variant / str(seed) / "run.json"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                run = json.load(handle)
            if run.get("variant") != variant or int(run.get("seed")) != seed:
                raise ValueError(f"Run identity mismatch in {path}")
            seeds_seen[variant].add(seed)
            for code in TASKS:
                collected[variant][code].append(run["task_metrics"][code])

    if not allow_incomplete:
        incomplete = {
            variant: sorted(set(SEEDS).difference(seeds))
            for variant, seeds in seeds_seen.items()
            if seeds != set(SEEDS)
        }
        if incomplete:
            raise FileNotFoundError(f"Missing three-seed runs: {incomplete}")

    summary: dict[str, dict[str, dict]] = {}
    for variant, tasks in collected.items():
        summary[variant] = {}
        for code, runs in tasks.items():
            if runs:
                summary[variant][code] = aggregate_seed_runs(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "paper_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "task", *[f"{metric}_mean" for metric in SUMMARY_METRICS], *[f"{metric}_std" for metric in SUMMARY_METRICS]]
        )
        for variant in VARIANTS:
            for code in TASKS:
                if code not in summary.get(variant, {}):
                    continue
                values = summary[variant][code]
                writer.writerow(
                    [variant, code]
                    + [values[metric]["mean"] for metric in SUMMARY_METRICS]
                    + [values[metric]["std"] for metric in SUMMARY_METRICS]
                )

    table_lines = [
        "# TQPT three-seed results",
        "",
        "All cells are mean ± sample standard deviation in percentage points.",
        "",
        "| Method | Task | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-Precision | Weighted-Recall | Weighted-F1 | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for code in TASKS:
            if code not in summary.get(variant, {}):
                continue
            row = summary[variant][code]
            table_lines.append(
                "| " + " | ".join(
                    [
                        variant,
                        code,
                        format_mean_std(row["accuracy"]),
                        format_mean_std(row["macro_precision"]),
                        format_mean_std(row["macro_recall"]),
                        format_mean_std(row["macro_f1"]),
                        format_mean_std(row["weighted_precision"]),
                        format_mean_std(row["weighted_recall"]),
                        format_mean_std(row["weighted_f1"]),
                        format_mean_std(row["invalid_label_rate"]),
                    ]
                ) + " |"
            )
    (output_dir / "paper_results.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    differences = [
        "# Ablation differences",
        "",
        "Values are TQPT mean minus ablation mean in percentage points.",
        "",
        "| Ablation | Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    if "TQPT" in summary:
        for ablation in ("TQPT_NT", "TQPT_NS"):
            for code in TASKS:
                if code not in summary["TQPT"] or code not in summary.get(ablation, {}):
                    continue
                base = summary["TQPT"][code]
                other = summary[ablation][code]
                differences.append(
                    f"| {ablation} | {code} | "
                    f"{100 * (base['accuracy']['mean'] - other['accuracy']['mean']):.2f} | "
                    f"{100 * (base['macro_f1']['mean'] - other['macro_f1']['mean']):.2f} | "
                    f"{100 * (base['weighted_f1']['mean'] - other['weighted_f1']['mean']):.2f} |"
                )
        if reference_file is not None and reference_file.is_file():
            with reference_file.open("r", encoding="utf-8") as handle:
                reference = json.load(handle)
            differences.extend(
                [
                    "",
                    "## Historical reference",
                    "",
                    "The historical thesis values are context only, not an acceptance oracle.",
                    "",
                    "| Task | Current Macro-F1 | Historical Macro-F1 | ΔMacro-F1 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for code in TASKS:
                if code not in summary["TQPT"] or code not in reference.get("metrics", {}):
                    continue
                current = summary["TQPT"][code]["macro_f1"]["mean"]
                historical = float(reference["metrics"][code]["macro_f1"])
                differences.append(
                    f"| {code} | {100 * current:.2f} | {100 * historical:.2f} | "
                    f"{100 * (current - historical):.2f} |"
                )
    (output_dir / "difference_report.md").write_text("\n".join(differences) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, execute, or summarize TQPT experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/tqpt.yaml"))
    run.add_argument("--variants", nargs="+", choices=VARIANTS)
    run.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    run.add_argument("--raw-root", type=Path)
    run.add_argument("--processed-root", type=Path)
    run.add_argument("--base-model", type=Path)
    run.add_argument("--extended-model", type=Path)
    run.add_argument("--extended-tokenizer", type=Path)
    run.add_argument("--llamafactory-dir", type=Path)
    run.add_argument("--runs-root", type=Path)
    run.add_argument("--results-root", type=Path)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--smoke-samples", type=int, default=64)
    run.add_argument("--smoke-steps", type=int, default=2)
    run.add_argument("--eval-samples", type=int)

    report = subparsers.add_parser("summarize")
    report.add_argument("--config", type=Path, default=Path("configs/tqpt.yaml"))
    report.add_argument("--results-root", type=Path)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--reference-file", type=Path, default=Path("configs/reference_results.json"))
    report.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        config = load_config(args.config)
        args.results_root = args.results_root or config_path(config, "results")
        args.output_dir = args.output_dir or args.results_root / "summary"
        if not args.reference_file.is_absolute():
            args.reference_file = Path(config["_repo_root"]) / args.reference_file
        result = summarize(
            args.results_root,
            args.output_dir,
            allow_incomplete=args.allow_incomplete,
            reference_file=args.reference_file,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    args = configure_run(args)
    commands = build_commands(args)
    if args.execute:
        run_commands(commands)
    else:
        print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

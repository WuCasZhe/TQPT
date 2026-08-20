from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, Mapping, MutableMapping, Optional, Sequence, TextIO, Union

from .config import config_path, load_config
from .prompts import build_stage2_prompt, split_instruction_and_traffic
from .registry import TASKS, instruction_output_to_code, load_labels


SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
DATA_SCHEMA_VERSION = 5
NORMALIZER_VERSION = 2
STAGE2_PROTOCOL = "trafficllm-official-boundary-tqpt-validation-v1"
STAGE2_VALIDATION_RATIO = 0.1


def iter_json_records(path: Union[Path, str]) -> Iterator[dict]:
    """Read a JSON array, one JSON object, or JSON Lines without guessing by suffix."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            value = json.load(handle)
            if not isinstance(value, list):
                raise TypeError(f"Expected JSON array in {source}")
            for record in value:
                if not isinstance(record, dict):
                    raise TypeError(f"Expected object records in {source}")
                yield record
            return

        if first == "{":
            # TrafficLLM files use JSON Lines despite the .json suffix. Parse
            # the first line before considering a potentially pretty-printed
            # single object, so a 200 MB file is never read into memory merely
            # to discover an "Extra data" error.
            first_line = handle.readline()
            try:
                first_value = json.loads(first_line)
            except json.JSONDecodeError:
                handle.seek(0)
                value = json.load(handle)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON object in {source}")
                yield value
                return
            if not isinstance(first_value, dict):
                raise TypeError(f"Expected object records in {source}")
            yield first_value
            for line_number, line in enumerate(handle, start=2):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected object at {source}:{line_number}")
                yield value
            return

        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {source}:{line_number}")
            yield value


def canonical_record_id(namespace: str, record: Mapping[str, object]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{namespace}\0{encoded}".encode("utf-8")).hexdigest()


def split_sizes(total: int) -> tuple[int, int, int]:
    if total < 0:
        raise ValueError("total cannot be negative")
    train = int(total * SPLIT_RATIOS[0])
    validation = int(total * SPLIT_RATIOS[1])
    return train, validation, total - train - validation


def stratified_assign(
    ids_by_label: Mapping[str, Sequence[str]], split_seed: int
) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Assign unique IDs to exact per-label 8:1:1 splits using a keyed hash order."""
    groups = {
        label: Counter(sample_ids)
        for label, sample_ids in ids_by_label.items()
    }
    return stratified_group_assign(groups, split_seed)


def stratified_group_assign(
    groups_by_label: Mapping[str, Mapping[str, int]], split_seed: int
) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Assign content groups without leakage while preserving exact record counts."""
    assignment: Dict[str, str] = {}
    counts: Dict[str, Dict[str, int]] = {split: {} for split in SPLITS}
    for label in sorted(groups_by_label):
        groups = {content_id: int(size) for content_id, size in groups_by_label[label].items()}
        if any(size <= 0 for size in groups.values()):
            raise ValueError(f"Content group sizes must be positive for label {label!r}")
        total = sum(groups.values())
        train_count, validation_count, test_count = split_sizes(total)
        targets = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        remaining = dict(targets)
        ranked = sorted(
            groups.items(),
            key=lambda item: (
                -item[1],
                hashlib.sha256(
                    f"{split_seed}\0{label}\0{item[0]}".encode("utf-8")
                ).digest(),
            ),
        )
        for content_id, size in ranked:
            candidates = [split for split in SPLITS if remaining[split] >= size]
            if not candidates:
                raise ValueError(
                    f"Cannot preserve content group {content_id} (size={size}) and exact 8:1:1 counts"
                )
            split = max(
                candidates,
                key=lambda name: (
                    remaining[name] / targets[name] if targets[name] else -1.0,
                    hashlib.sha256(
                        f"{split_seed}\0{label}\0{content_id}\0{name}".encode("utf-8")
                    ).digest(),
                ),
            )
            assignment[content_id] = split
            remaining[split] -= size
        if any(remaining.values()):
            raise AssertionError(f"Exact grouped split allocation failed for {label}: {remaining}")
        counts["train"][label] = train_count
        counts["validation"][label] = validation_count
        counts["test"][label] = test_count
    return assignment, counts


def _open_split_outputs(root: Path) -> Dict[str, TextIO]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        split: (root / f"{split}.jsonl.tmp").open("w", encoding="utf-8")
        for split in SPLITS
    }


def _commit_split_outputs(root: Path, handles: MutableMapping[str, TextIO]) -> None:
    for split, handle in handles.items():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        (root / f"{split}.jsonl.tmp").replace(root / f"{split}.jsonl")


def _abort_split_outputs(root: Path, handles: MutableMapping[str, TextIO]) -> None:
    for handle in handles.values():
        if not handle.closed:
            handle.close()
    for split in SPLITS:
        (root / f"{split}.jsonl.tmp").unlink(missing_ok=True)


def _task_source_paths(raw_root: Path, code: str) -> tuple[Path, Path]:
    spec = TASKS[code]
    directory = raw_root / spec.dataset_dir
    return directory / spec.train_file, directory / spec.test_file


def prepare_task(
    raw_root: Path,
    output_root: Path,
    code: str,
    split_seed: int,
    *,
    normalize: bool = True,
) -> dict:
    """Keep TrafficLLM's released test set fixed and split validation from train."""
    labels = load_labels(raw_root, code)
    train_source, test_source = _task_source_paths(raw_root, code)
    groups_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    for record in iter_json_records(train_source):
        label = str(record.get("output", ""))
        if label not in labels:
            raise ValueError(f"Unknown {code} label {label!r} in {train_source}")
        groups_by_label[label][canonical_record_id(code, record)] += 1

    assignment: Dict[str, str] = {}
    planned_counts = {"train": {}, "validation": {}}
    for label in sorted(groups_by_label):
        groups = groups_by_label[label]
        total = sum(groups.values())
        validation_count = int(total * STAGE2_VALIDATION_RATIO)
        targets = {"train": total - validation_count, "validation": validation_count}
        remaining = dict(targets)
        ranked = sorted(
            groups.items(),
            key=lambda item: (
                -item[1],
                hashlib.sha256(
                    f"{split_seed}\0{label}\0{item[0]}".encode("utf-8")
                ).digest(),
            ),
        )
        for content_id, size in ranked:
            candidates = [name for name in targets if remaining[name] >= size]
            if not candidates:
                raise ValueError(
                    f"Cannot preserve content group {content_id} (size={size}) "
                    "and exact train/validation counts"
                )
            destination = max(
                candidates,
                key=lambda name: (
                    remaining[name] / targets[name] if targets[name] else -1.0,
                    hashlib.sha256(
                        f"{split_seed}\0{label}\0{content_id}\0{name}".encode("utf-8")
                    ).digest(),
                ),
            )
            assignment[content_id] = destination
            remaining[destination] -= size
        if any(remaining.values()):
            raise AssertionError(
                f"Exact train/validation allocation failed for {label}: {remaining}"
            )
        for destination, count in targets.items():
            planned_counts[destination][label] = count

    task_root = output_root / "stage2" / code.lower()
    handles = _open_split_outputs(task_root)
    occurrences: Counter[str] = Counter()
    contents_by_split = {split: Counter() for split in SPLITS}
    actual_counts = {split: Counter() for split in SPLITS}
    try:
        for source_split, source in (("train", train_source), ("test", test_source)):
            for record in iter_json_records(source):
                label = str(record.get("output", ""))
                if label not in labels:
                    raise ValueError(f"Unknown {code} label {label!r} in {source}")
                content_id = canonical_record_id(code, record)
                occurrence = occurrences[content_id]
                occurrences[content_id] += 1
                sample_id = hashlib.sha256(
                    f"{content_id}\0{occurrence}".encode("utf-8")
                ).hexdigest()
                task_instruction, traffic = split_instruction_and_traffic(str(record["instruction"]))
                prepared = {
                    "instruction": build_stage2_prompt(
                        task_instruction, traffic, normalize=normalize
                    ),
                    "input": "",
                    "output": label,
                    "task_instruction": task_instruction,
                    "task_code": code,
                    "label": label,
                    "sample_id": sample_id,
                    "content_id": content_id,
                    "source_split": source_split,
                }
                if not normalize:
                    prepared["traffic"] = traffic
                split = "test" if source_split == "test" else assignment[content_id]
                handles[split].write(json.dumps(prepared, ensure_ascii=False) + "\n")
                actual_counts[split][label] += 1
                contents_by_split[split][content_id] += 1
    except Exception:
        _abort_split_outputs(task_root, handles)
        raise
    else:
        _commit_split_outputs(task_root, handles)

    actual = {split: dict(sorted(counter.items())) for split, counter in actual_counts.items()}
    if actual["train"] != planned_counts["train"]:
        raise AssertionError(f"Written train counts differ from plan for {code}")
    if actual["validation"] != planned_counts["validation"]:
        raise AssertionError(f"Written validation counts differ from plan for {code}")
    input_samples = sum(sum(counter.values()) for counter in actual_counts.values())
    unique_contents = len(set().union(*(set(counter) for counter in contents_by_split.values())))
    duplicate_count = input_samples - unique_contents
    return {
        "task_code": code,
        "protocol": STAGE2_PROTOCOL,
        "labels": labels,
        "input_samples": input_samples,
        "unique_contents": unique_contents,
        "duplicate_occurrences_preserved": duplicate_count,
        "validation_source": "held-out-from-released-train",
        "validation_ratio_within_released_train": STAGE2_VALIDATION_RATIO,
        "split_counts": actual,
        "source_files": {
            "released_train": str(train_source.relative_to(raw_root)),
            "released_test": str(test_source.relative_to(raw_root)),
        },
    }


def prepare_router(raw_root: Path, output_root: Path, split_seed: int) -> dict:
    source = raw_root / "instructions" / "instruction.json"
    records: list[dict] = []
    groups_by_code: Dict[str, Counter[str]] = defaultdict(Counter)
    ignored = 0
    duplicate_count = 0
    for record in iter_json_records(source):
        try:
            code = instruction_output_to_code(str(record.get("output", "")))
        except KeyError:
            ignored += 1
            continue
        normalized = {
            "instruction": str(record.get("instruction", "")).strip(),
            "input": "",
            "output": code,
            "task_code": code,
        }
        if not normalized["instruction"]:
            raise ValueError(f"Empty router instruction in {source}")
        content_id = canonical_record_id("ROUTER", normalized)
        if groups_by_code[code][content_id]:
            duplicate_count += 1
        groups_by_code[code][content_id] += 1
        normalized["content_id"] = content_id
        records.append(normalized)

    assignment, planned_counts = stratified_group_assign(groups_by_code, split_seed)
    router_root = output_root / "stage1"
    handles = _open_split_outputs(router_root)
    actual_counts = {split: Counter() for split in SPLITS}
    occurrences: Counter[str] = Counter()
    try:
        for record in records:
            content_id = record["content_id"]
            occurrence = occurrences[content_id]
            occurrences[content_id] += 1
            record["sample_id"] = hashlib.sha256(
                f"{content_id}\0{occurrence}".encode("utf-8")
            ).hexdigest()
            split = assignment[content_id]
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            actual_counts[split][record["task_code"]] += 1
    except Exception:
        _abort_split_outputs(router_root, handles)
        raise
    else:
        _commit_split_outputs(router_root, handles)

    actual = {split: dict(sorted(counter.items())) for split, counter in actual_counts.items()}
    if actual != planned_counts:
        raise AssertionError("Written router split counts differ from plan")
    return {
        "source_file": str(source.relative_to(raw_root)),
        "input_samples": len(records),
        "unique_contents": sum(len(groups) for groups in groups_by_code.values()),
        "duplicate_occurrences_grouped": duplicate_count,
        "ignored_other_tasks": ignored,
        "split_counts": actual,
    }


def write_dataset_info(output_root: Path) -> Path:
    registry: dict[str, dict] = {}
    for split in SPLITS:
        registry[f"tqpt_stage1_{split}"] = {
            "file_name": f"stage1/{split}.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
        for code in TASKS:
            registry[f"tqpt_{code.lower()}_{split}"] = {
                "file_name": f"stage2/{code.lower()}/{split}.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            }
    path = output_root / "dataset_info.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
    return path


def verify_no_overlap(output_root: Path) -> dict:
    report: dict[str, dict] = {}
    groups = {"stage1": output_root / "stage1"}
    groups.update({code: output_root / "stage2" / code.lower() for code in TASKS})
    for name, directory in groups.items():
        split_ids: Dict[str, set[str]] = {}
        split_contents: Dict[str, set[str]] = {}
        for split in SPLITS:
            split_ids[split] = set()
            split_contents[split] = set()
            for record in iter_json_records(directory / f"{split}.jsonl"):
                split_ids[split].add(str(record["sample_id"]))
                split_contents[split].add(str(record["content_id"]))
        sample_intersections = {
            "train_validation": len(split_ids["train"] & split_ids["validation"]),
            "train_test": len(split_ids["train"] & split_ids["test"]),
            "validation_test": len(split_ids["validation"] & split_ids["test"]),
        }
        content_intersections = {
            "train_validation": len(split_contents["train"] & split_contents["validation"]),
            "train_test": len(split_contents["train"] & split_contents["test"]),
            "validation_test": len(split_contents["validation"] & split_contents["test"]),
        }
        if name == "stage1":
            if any(sample_intersections.values()) or any(content_intersections.values()):
                raise AssertionError(
                    f"Router split leakage detected: samples={sample_intersections}, "
                    f"contents={content_intersections}"
                )
            expectation = "disjoint-router-splits"
        else:
            if any(sample_intersections.values()):
                raise AssertionError(
                    f"Stage-two sample leakage detected in {name}: {sample_intersections}"
                )
            if content_intersections["train_validation"]:
                raise AssertionError(
                    f"Stage-two train/validation content leakage detected in {name}: "
                    f"{content_intersections['train_validation']}"
                )
            # Exact contents duplicated by the released source across train and
            # test remain reported here; the official test boundary is not rewritten.
            expectation = "held-out-validation; released-test-boundary-preserved"
        report[name] = {
            "expectation": expectation,
            "sizes": {split: len(ids) for split, ids in split_ids.items()},
            "sample_id_intersections": sample_intersections,
            "content_id_intersections": content_intersections,
        }
    return report


def prepare_all(raw_root: Path, output_root: Path, split_seed: int, *, force: bool = False) -> dict:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to rebuild")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = {
        code: prepare_task(raw_root, output_root, code, split_seed)
        for code in TASKS
    }
    router = prepare_router(raw_root, output_root, split_seed)
    dataset_info = write_dataset_info(output_root)
    overlap = verify_no_overlap(output_root)
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "split_seed": split_seed,
        "split_ratios": list(SPLIT_RATIOS),
        "stage2_protocol": STAGE2_PROTOCOL,
        "stage2_validation_ratio": STAGE2_VALIDATION_RATIO,
        "task_name_to_code": {spec.name: code for code, spec in TASKS.items()},
        "router": router,
        "tasks": tasks,
        "overlap_verification": overlap,
        "dataset_info": str(dataset_info.relative_to(output_root)),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary.replace(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare official-boundary stage-two data and deterministic router splits"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/tqpt.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    args.raw_root = args.raw_root or config_path(config, "raw_data")
    args.output_root = args.output_root or config_path(config, "processed_data")
    args.split_seed = args.split_seed if args.split_seed is not None else int(config["project"]["split_seed"])
    manifest = prepare_all(args.raw_root, args.output_root, args.split_seed, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

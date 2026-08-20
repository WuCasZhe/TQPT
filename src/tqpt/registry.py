from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union


@dataclass(frozen=True)
class TaskSpec:
    code: str
    name: str
    dataset_dir: str
    train_file: str
    test_file: str
    label_file: str
    expected_labels: int
    instruction_outputs: tuple[str, ...]


TASKS: Mapping[str, TaskSpec] = {
    "EVD": TaskSpec(
        code="EVD",
        name="Encrypted VPN Detection",
        dataset_dir="iscx-vpn-2016",
        train_file="iscx-vpn-2016_detection_packet_train.json",
        test_file="iscx-vpn-2016_detection_packet_test.json",
        label_file="iscx-vpn-2016_label.json",
        expected_labels=14,
        instruction_outputs=("Encrypted VPN Detection", "VPN Detection", "EVD"),
    ),
    "AAD": TaskSpec(
        code="AAD",
        name="APT Attack Detection",
        dataset_dir="dapt-2020",
        train_file="dapt-2020_detection_packet_train.json",
        test_file="dapt-2020_detection_packet_test.json",
        label_file="dapt-2020_label.json",
        expected_labels=2,
        instruction_outputs=("APT Attack Detection", "APT Detection", "AAD", "APT"),
    ),
    "CD": TaskSpec(
        code="CD",
        name="Concept Drift",
        dataset_dir="app53-2023",
        train_file="app53-2023_detection_packet_train.json",
        test_file="app53-2023_detection_packet_test.json",
        label_file="app53-2023_label.json",
        expected_labels=54,
        instruction_outputs=("Concept Drift", "Concept Drift Detection", "CD"),
    ),
}


def _normalized_alias(text: str) -> str:
    return " ".join(text.strip().casefold().replace("_", " ").split())


INSTRUCTION_OUTPUT_TO_CODE: Dict[str, str] = {
    _normalized_alias(alias): code
    for code, spec in TASKS.items()
    for alias in spec.instruction_outputs
}


def instruction_output_to_code(text: str) -> str:
    """Map TrafficLLM's task names (including APT/AAD aliases) to one code."""
    key = _normalized_alias(text)
    try:
        return INSTRUCTION_OUTPUT_TO_CODE[key]
    except KeyError as exc:
        raise KeyError(f"Unsupported instruction task name: {text!r}") from exc


def get_task(code: str) -> TaskSpec:
    normalized = code.strip().upper()
    try:
        return TASKS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown task code {code!r}; expected one of {sorted(TASKS)}") from exc


def load_labels(raw_root: Union[Path, str], code: str) -> Dict[str, int]:
    spec = get_task(code)
    path = Path(raw_root) / spec.dataset_dir / spec.label_file
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, dict):
        raise TypeError(f"Label registry must be a JSON object: {path}")
    labels = {str(label): int(index) for label, index in labels.items()}
    if len(labels) != spec.expected_labels:
        raise ValueError(
            f"{code} expects {spec.expected_labels} labels, found {len(labels)} in {path}"
        )
    expected_ids = list(range(spec.expected_labels))
    if sorted(labels.values()) != expected_ids:
        raise ValueError(f"{code} label IDs must be contiguous 0..{spec.expected_labels - 1}")
    return labels


def validate_task_codes(codes: Iterable[str]) -> tuple[str, ...]:
    result = tuple(code.strip().upper() for code in codes)
    unknown = sorted(set(result).difference(TASKS))
    if unknown:
        raise ValueError(f"Unknown task codes: {unknown}")
    return result

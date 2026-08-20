from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

from .prompts import normalize_generated_label


INVALID_LABEL = "__INVALID__"


def _safe_div(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    predictions: Sequence[str], targets: Sequence[str], label_names: Sequence[str]
) -> dict:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have identical lengths")
    labels = list(label_names)
    if len(labels) != len(set(labels)):
        raise ValueError("label_names contains duplicates")
    label_set = set(labels)
    normalized_predictions = [normalize_generated_label(value) for value in predictions]
    normalized_targets = [normalize_generated_label(value) for value in targets]
    unknown_targets = sorted(set(normalized_targets).difference(label_set))
    if unknown_targets:
        raise ValueError(f"Targets not present in label registry: {unknown_targets[:5]}")

    canonical_predictions = [value if value in label_set else INVALID_LABEL for value in normalized_predictions]
    columns = labels + [INVALID_LABEL]
    column_index = {label: index for index, label in enumerate(columns)}
    row_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in columns] for _ in labels]
    for target, prediction in zip(normalized_targets, canonical_predictions):
        matrix[row_index[target]][column_index[prediction]] += 1

    per_class: dict[str, dict[str, Union[float, int]]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    correct = 0
    for label in labels:
        index = row_index[label]
        tp = matrix[index][column_index[label]]
        fp = sum(matrix[row][column_index[label]] for row in range(len(labels)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(columns)) if column != column_index[label])
        support = sum(matrix[index])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        correct += tp

    total = len(targets)
    invalid = sum(prediction == INVALID_LABEL for prediction in canonical_predictions)
    weighted_f1 = _safe_div(sum(f1 * support for f1, support in zip(f1s, supports)), total)
    weighted_precision = _safe_div(
        sum(precision * support for precision, support in zip(precisions, supports)), total
    )
    weighted_recall = _safe_div(
        sum(recall * support for recall, support in zip(recalls, supports)), total
    )
    macro_precision = _safe_div(sum(precisions), len(labels))
    macro_recall = _safe_div(sum(recalls), len(labels))
    macro_f1 = _safe_div(sum(f1s), len(labels))
    classification_report = {
        label: {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1-score": float(values["f1"]),
            "support": int(values["support"]),
        }
        for label, values in per_class.items()
    }
    classification_report.update(
        {
            "accuracy": _safe_div(correct, total),
            "macro avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1-score": macro_f1,
                "support": total,
            },
            "weighted avg": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1-score": weighted_f1,
                "support": total,
            },
        }
    )
    return {
        "samples": total,
        "accuracy": _safe_div(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "invalid_label_rate": _safe_div(invalid, total),
        "invalid_label_count": invalid,
        "labels": labels,
        "confusion_matrix_columns": columns,
        "confusion_matrix": matrix,
        "classification_report": classification_report,
        "per_class": per_class,
    }


def prediction_rows(
    sample_ids: Sequence[str],
    task_codes: Sequence[str],
    route_outputs: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
) -> list[dict]:
    lengths = {len(sample_ids), len(task_codes), len(route_outputs), len(predictions), len(targets)}
    if len(lengths) != 1:
        raise ValueError("All per-sample sequences must have identical lengths")
    rows = []
    for sample_id, task_code, route, prediction, target in zip(
        sample_ids, task_codes, route_outputs, predictions, targets
    ):
        normalized = normalize_generated_label(prediction)
        rows.append(
            {
                "sample_id": sample_id,
                "task_code": task_code,
                "route_output": route,
                "target": target,
                "raw_prediction": prediction,
                "normalized_prediction": normalized,
                "correct": normalized == target,
            }
        )
    return rows


def write_evaluation(output_dir: Union[Path, str], metrics: Mapping, rows: Iterable[Mapping]) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, ensure_ascii=False, indent=2)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


SUMMARY_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "invalid_label_rate",
)


def aggregate_seed_runs(run_metrics: Sequence[Mapping[str, float]]) -> dict:
    if not run_metrics:
        raise ValueError("At least one run is required")
    summary: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(run[metric]) for run in run_metrics]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    summary["seeds"] = len(run_metrics)
    return summary


def format_mean_std(value: Mapping[str, float], scale: float = 100.0) -> str:
    mean = float(value["mean"]) * scale
    std = float(value["std"]) * scale
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("Cannot format non-finite experiment metrics")
    return f"{mean:.2f} ± {std:.2f}"

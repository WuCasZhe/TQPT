from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from .config import require_local_model_snapshot
from .data import iter_json_records
from .metrics import classification_metrics, prediction_rows, write_evaluation
from .prompts import parse_route_code
from .registry import TASKS, load_labels
from .training import (
    select_generation_auto_class,
    sha256_file,
    strip_prefix_state_dict,
    validate_adapter_metadata,
)


def batches(values: Sequence, size: int) -> Iterator[Sequence]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _model_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine model device")


def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    results: list[str] = []
    model.eval()
    device = _model_device(model)
    for prompt_batch in batches(list(prompts), batch_size):
        rendered = [
            tokenizer.build_prompt(prompt, history=[]) if hasattr(tokenizer, "build_prompt") else prompt
            for prompt in prompt_batch
        ]
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        new_tokens = generated[:, input_length:]
        results.extend(
            tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return results


def load_qlora_model(model_path: Path, adapter_path: Path):
    require_local_model_snapshot(model_path)
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("QLoRA inference requires the locked GPU dependencies") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_class = select_generation_auto_class(
        config,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = model_class.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    return model.eval(), tokenizer


def load_prefix_base(
    model_path: Path,
    tokenizer_path: Path,
    pre_seq_len: int,
    prefix_projection: bool,
):
    require_local_model_snapshot(model_path)
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Prefix inference requires the locked GPU dependencies") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.pre_seq_len = pre_seq_len
    config.prefix_projection = prefix_projection
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.half().cuda().eval()
    model.transformer.prefix_encoder.float()
    return model, tokenizer


def activate_prefix(model, checkpoint: Path) -> None:
    import torch

    state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu")
    model.transformer.prefix_encoder.load_state_dict(strip_prefix_state_dict(state), strict=True)
    model.transformer.prefix_encoder.float()


def _load_test_records(processed_root: Path, max_samples: int) -> list[dict]:
    records: list[dict] = []
    for code in TASKS:
        path = processed_root / "stage2" / code.lower() / "test.jsonl"
        selected = list(islice(iter_json_records(path), max_samples))
        for record in selected:
            if record.get("task_code") != code:
                raise ValueError(f"Prepared test record has wrong task code in {path}")
        records.extend(selected)
    return records


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_two_stage(args: argparse.Namespace) -> dict:
    records = _load_test_records(args.processed_root, args.max_samples_per_task)
    router_metadata = validate_adapter_metadata(
        args.router_adapter,
        adapter_type="qlora-router",
    )
    if sorted(router_metadata.get("task_codes", [])) != sorted(TASKS):
        raise ValueError("Router adapter task registry does not match EVD/AAD/CD")
    router_model, router_tokenizer = load_qlora_model(args.model, args.router_adapter)
    route_raw = generate_texts(
        router_model,
        router_tokenizer,
        [str(record["task_instruction"]) for record in records],
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_router,
    )
    route_codes = [parse_route_code(value) for value in route_raw]
    del router_model
    _release_cuda_cache()

    predictions = [""] * len(records)
    routed_indices: dict[str, list[int]] = defaultdict(list)
    for index, code in enumerate(route_codes):
        if code is not None:
            routed_indices[code].append(index)

    prefix_model, prefix_tokenizer = load_prefix_base(
        args.model,
        args.tokenizer,
        args.pre_seq_len,
        args.prefix_projection,
    )
    prefix_metadata: dict[str, dict] = {}
    for code in TASKS:
        adapter = args.prefix_root / code.lower()
        spec = TASKS[code]
        label_path = args.raw_root / spec.dataset_dir / spec.label_file
        prefix_metadata[code] = validate_adapter_metadata(
            adapter,
            adapter_type="prefix",
            task_code=code,
            label_registry_sha256=sha256_file(label_path),
        )
        indices = routed_indices.get(code, [])
        if not indices:
            continue
        activate_prefix(prefix_model, adapter)
        outputs = generate_texts(
            prefix_model,
            prefix_tokenizer,
            [str(records[index]["instruction"]) for index in indices],
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens_label,
        )
        for index, output in zip(indices, outputs):
            predictions[index] = output
    del prefix_model
    _release_cuda_cache()

    return _write_run_results(
        args,
        records,
        route_raw,
        predictions,
        {
            "router_adapter": str(args.router_adapter.resolve()),
            "router_metadata": router_metadata,
            "prefix_metadata": prefix_metadata,
        },
    )


def evaluate_joint(args: argparse.Namespace) -> dict:
    records = _load_test_records(args.processed_root, args.max_samples_per_task)
    metadata = validate_adapter_metadata(
        args.joint_adapter,
        adapter_type="qlora-classifier",
    )
    if sorted(metadata.get("task_codes", [])) != sorted(TASKS):
        raise ValueError("Joint adapter task registry does not match EVD/AAD/CD")
    model, tokenizer = load_qlora_model(args.model, args.joint_adapter)
    predictions = generate_texts(
        model,
        tokenizer,
        [str(record["instruction"]) for record in records],
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_label,
    )
    del model
    _release_cuda_cache()
    route_outputs = [str(record["task_code"]) for record in records]
    return _write_run_results(
        args,
        records,
        route_outputs,
        predictions,
        {
            "joint_adapter": str(args.joint_adapter.resolve()),
            "joint_metadata": metadata,
        },
    )


def _write_run_results(
    args: argparse.Namespace,
    records: Sequence[Mapping],
    route_outputs: Sequence[str],
    predictions: Sequence[str],
    adapter_details: Mapping,
) -> dict:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    task_metrics: dict[str, dict] = {}
    for code in TASKS:
        indices = [index for index, record in enumerate(records) if record["task_code"] == code]
        task_predictions = [predictions[index] for index in indices]
        targets = [str(records[index]["output"]) for index in indices]
        labels = load_labels(args.raw_root, code)
        ordered_labels = sorted(labels, key=labels.get)
        metrics = classification_metrics(task_predictions, targets, ordered_labels)
        metrics.update({"variant": args.variant, "seed": args.seed, "task_code": code})
        rows = prediction_rows(
            [str(records[index]["sample_id"]) for index in indices],
            [code] * len(indices),
            [route_outputs[index] for index in indices],
            task_predictions,
            targets,
        )
        write_evaluation(output_root / code.lower(), metrics, rows)
        task_metrics[code] = metrics

    route_targets = [str(record["task_code"]) for record in records]
    route_metrics = classification_metrics(list(route_outputs), route_targets, list(TASKS))
    run = {
        "variant": args.variant,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_samples_per_task": args.max_samples_per_task,
        },
        "router_metrics": route_metrics,
        "task_metrics": task_metrics,
        **dict(adapter_details),
    }
    with (output_root / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic TQPT inference and evaluation")
    parser.add_argument("--variant", choices=("TQPT", "TQPT_NT", "TQPT_NS"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--router-adapter", type=Path)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--joint-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-seq-len", type=int, default=128)
    parser.add_argument("--prefix-projection", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens-router", type=int, default=8)
    parser.add_argument("--max-new-tokens-label", type=int, default=32)
    parser.add_argument("--max-samples-per-task", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.variant == "TQPT_NS":
        if args.joint_adapter is None:
            raise SystemExit("--joint-adapter is required for TQPT_NS")
        result = evaluate_joint(args)
    else:
        if args.router_adapter is None or args.prefix_root is None:
            raise SystemExit("--router-adapter and --prefix-root are required for two-stage variants")
        result = evaluate_two_stage(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

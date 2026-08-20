from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import require_local_model_snapshot
from .data import STAGE2_PROTOCOL, iter_json_records
from .registry import TASKS, get_task
from .tokenizer import (
    ensure_chatglm2_model_remote_code,
    ensure_chatglm2_tokenizer_remote_code,
)
from .training import (
    StableCausalLossTrainerMixin,
    build_causal_example,
    freeze_except_prefix,
    prefix_state_dict,
    reset_prefix_encoder_parameters,
    sha256_file,
    tokenizer_fingerprint,
    write_adapter_metadata,
)


def _validate_dataset_labels(path: Path, valid_labels: set[str], task_code: str) -> None:
    for index, record in enumerate(iter_json_records(path), start=1):
        if record.get("task_code") != task_code:
            raise ValueError(f"Wrong task_code in {path}:{index}")
        if str(record.get("output")) not in valid_labels:
            raise ValueError(f"Unknown label in {path}:{index}: {record.get('output')!r}")


def train(args: argparse.Namespace) -> dict:
    require_local_model_snapshot(args.model)
    ensure_chatglm2_model_remote_code(args.model)
    ensure_chatglm2_tokenizer_remote_code(args.tokenizer)
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Install the locked GPU training dependencies before Prefix-Tuning") from exc

    spec = get_task(args.task)
    task_code = spec.code
    with args.label_file.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, dict) or len(labels) != spec.expected_labels:
        raise ValueError(f"Invalid label registry for {task_code}: {args.label_file}")
    valid_labels = set(labels)
    _validate_dataset_labels(args.train_file, valid_labels, task_code)
    _validate_dataset_labels(args.validation_file, valid_labels, task_code)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection
    model = AutoModel.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    input_rows = int(model.get_input_embeddings().weight.shape[0])
    output_layer = model.get_output_embeddings()
    if output_layer is None:
        output_layer = model.transformer.output_layer
    output_rows = int(output_layer.weight.shape[0])
    max_token_id = max(tokenizer.get_vocab().values())
    if input_rows <= max_token_id or output_rows <= max_token_id:
        raise ValueError(
            f"Model vocabulary matrices ({input_rows}, {output_rows}) do not cover tokenizer ID {max_token_id}"
        )
    trainable = freeze_except_prefix(model)
    model = model.half()
    model.transformer.prefix_encoder.float()
    reset_prefix_encoder_parameters(model.transformer.prefix_encoder)
    if hasattr(model, "config"):
        model.config.use_cache = False

    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )

    def preprocess(batch):
        result = {"input_ids": [], "labels": []}
        for prompt, answer in zip(batch["instruction"], batch["output"]):
            encoded = build_causal_example(
                tokenizer,
                prompt=str(prompt),
                answer=str(answer),
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
            )
            result["input_ids"].append(encoded["input_ids"])
            result["labels"].append(encoded["labels"])
        return result

    columns = raw["train"].column_names
    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=columns,
        num_proc=args.preprocessing_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Tokenizing {task_code} Prefix-Tuning data",
    )
    if args.max_samples is not None:
        for split in ("train", "validation"):
            tokenized[split] = tokenized[split].select(
                range(min(args.max_samples, len(tokenized[split])))
            )

    metadata = {
        "schema_version": 1,
        "adapter_type": "prefix",
        "data_protocol": STAGE2_PROTOCOL,
        "variant": args.variant,
        "task_code": task_code,
        "seed": args.seed,
        "base_model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": tokenizer_fingerprint(args.tokenizer),
        "label_registry_sha256": sha256_file(args.label_file),
        "label_count": len(labels),
        "pre_seq_len": args.pre_seq_len,
        "prefix_projection": args.prefix_projection,
        "trainable_parameter_names": trainable,
    }

    class PrefixTrainer(StableCausalLossTrainerMixin, Seq2SeqTrainer):
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(prefix_state_dict(self.model), destination / "pytorch_model.bin")
            self.tokenizer.save_pretrained(destination)
            torch.save(self.args, destination / "training_args.bin")
            write_adapter_metadata(destination, metadata)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=True,
        logging_nan_inf_filter=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )
    trainer = PrefixTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_metrics("train", result.metrics)
    metadata.update(
        {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "eval_loss": eval_metrics.get("eval_loss"),
        }
    )
    write_adapter_metadata(args.output_dir, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one native ChatGLM2 PrefixEncoder")
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--variant", choices=("TQPT", "TQPT_NT"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--pre-seq-len", type=int, default=128)
    parser.add_argument("--prefix-projection", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=4000)
    parser.add_argument("--save-steps", type=int, default=4000)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

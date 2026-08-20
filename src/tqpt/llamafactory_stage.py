from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

from .config import require_local_model_snapshot
from .data import STAGE2_PROTOCOL
from .registry import TASKS
from .tokenizer import (
    ensure_chatglm2_model_remote_code,
    ensure_chatglm2_tokenizer_remote_code,
)
from .training import (
    StableCausalLossTrainerMixin,
    assert_only_lora_trainable,
    build_causal_example,
    select_generation_auto_class,
    tokenizer_fingerprint,
    write_adapter_metadata,
)


LLAMAFACTORY_COMMIT = "baf2e4e825a61ffabef2b9f86d654f73ace8d120"


def activate_llamafactory(source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    package_root = source_dir / "src"
    if not (package_root / "llmtuner").is_dir():
        raise FileNotFoundError(
            f"LLaMA-Factory v0.1.0 source not found at {source_dir}. "
            "Run scripts/bootstrap_llamafactory.sh first."
        )
    if (source_dir / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != LLAMAFACTORY_COMMIT:
            raise RuntimeError(
                f"LLaMA-Factory revision mismatch: expected {LLAMAFACTORY_COMMIT}, got {revision}"
            )
    sys.path.insert(0, str(package_root))


def _dataset_files(processed_root: Path, mode: str) -> dict[str, Union[list[str], str]]:
    if mode == "router":
        return {
            "train": str(processed_root / "stage1" / "train.jsonl"),
            "validation": str(processed_root / "stage1" / "validation.jsonl"),
        }
    if mode == "joint":
        return {
            "train": [
                str(processed_root / "stage2" / code.lower() / "train.jsonl")
                for code in TASKS
            ],
            "validation": [
                str(processed_root / "stage2" / code.lower() / "validation.jsonl")
                for code in TASKS
            ],
        }
    raise ValueError(f"Unsupported LLaMA-Factory stage mode: {mode}")


def train(args: argparse.Namespace) -> dict:
    require_local_model_snapshot(args.model)
    ensure_chatglm2_model_remote_code(args.model)
    ensure_chatglm2_tokenizer_remote_code(args.model)
    activate_llamafactory(args.llamafactory_dir)
    try:
        import torch
        from datasets import load_dataset
        from peft.utils import set_peft_model_state_dict
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainingArguments,
            set_seed,
        )
        from transformers.modeling_utils import unwrap_model
        from llmtuner.hparams import FinetuningArguments, ModelArguments
        import llmtuner.tuner.core.loader as llamafactory_loader
        from llmtuner.tuner.sft.trainer import Seq2SeqPeftTrainer
    except ImportError as exc:
        raise RuntimeError("Locked GPU dependencies and LLaMA-Factory v0.1.0 are required") from exc

    set_seed(args.seed)
    model_args = ModelArguments(
        model_name_or_path=str(args.model),
        use_fast_tokenizer=False,
        padding_side="left",
        quantization_bit=4,
        quantization_type="nf4",
        double_quantization=True,
    )
    model_args.compute_dtype = torch.float16
    finetuning_args = FinetuningArguments(
        finetuning_type="lora",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target="query_key_value",
    )
    loader_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    llamafactory_loader.AutoModelForCausalLM = select_generation_auto_class(
        loader_config,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )
    model, tokenizer = llamafactory_loader.load_model_and_tokenizer(
        model_args,
        finetuning_args,
        is_trainable=True,
        stage="sft",
    )
    trainable = assert_only_lora_trainable(model)
    if hasattr(model, "config"):
        model.config.use_cache = False

    files = _dataset_files(args.processed_root, args.mode)
    raw = load_dataset("json", data_files=files)

    def preprocess(batch):
        examples = {"input_ids": [], "labels": []}
        for prompt, answer in zip(batch["instruction"], batch["output"]):
            encoded = build_causal_example(
                tokenizer,
                prompt=str(prompt),
                answer=str(answer),
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
            )
            examples["input_ids"].append(encoded["input_ids"])
            examples["labels"].append(encoded["labels"])
        return examples

    columns = raw["train"].column_names
    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=columns,
        num_proc=args.preprocessing_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Tokenizing {args.mode} QLoRA data",
    )
    if args.max_samples is not None:
        tokenized["train"] = tokenized["train"].select(
            range(min(args.max_samples, len(tokenized["train"])))
        )
        tokenized["validation"] = tokenized["validation"].select(
            range(min(args.max_samples, len(tokenized["validation"])))
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_smoothing_factor=0.0,
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

    class ResumableLoraTrainer(StableCausalLossTrainerMixin, Seq2SeqPeftTrainer):
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            checkpoint = Path(resume_from_checkpoint)
            adapter_bin = checkpoint / "adapter_model.bin"
            adapter_safe = checkpoint / "adapter_model.safetensors"
            if adapter_bin.is_file() or adapter_safe.is_file():
                active_model = unwrap_model(model or self.model)
                if adapter_safe.is_file():
                    from safetensors.torch import load_file

                    state = load_file(str(adapter_safe), device="cpu")
                else:
                    state = torch.load(adapter_bin, map_location="cpu")
                set_peft_model_state_dict(active_model, state, adapter_name="default")
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _load_best_model(self):
            self._load_from_checkpoint(self.state.best_model_checkpoint)

    trainer = ResumableLoraTrainer(
        finetuning_args=finetuning_args,
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

    metadata = {
        "schema_version": 1,
        "adapter_type": "qlora-router" if args.mode == "router" else "qlora-classifier",
        "mode": args.mode,
        "variant": args.variant,
        "seed": args.seed,
        "base_model": str(args.model.resolve()),
        "tokenizer_sha256": tokenizer_fingerprint(args.model),
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quantization": True,
        },
        "lora": {
            "target": "query_key_value",
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "task_codes": sorted(TASKS),
        "trainable_parameter_names": trainable,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "eval_loss": eval_metrics.get("eval_loss"),
    }
    if args.mode == "joint":
        metadata["data_protocol"] = STAGE2_PROTOCOL
    write_adapter_metadata(args.output_dir, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TQPT QLoRA stages using LLaMA-Factory v0.1.0's NF4 loader"
    )
    parser.add_argument("--mode", choices=("router", "joint"), required=True)
    parser.add_argument("--variant", choices=("TQPT", "TQPT_NT", "TQPT_NS"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llamafactory-dir", type=Path, default=Path("third_party/LLaMA-Factory"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
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

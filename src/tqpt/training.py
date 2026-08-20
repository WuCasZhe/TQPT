from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Optional, Union


PREFIX_KEY = "transformer.prefix_encoder."


def reset_prefix_encoder_parameters(prefix_encoder) -> list[str]:
    """Initialize checkpoint-missing PrefixEncoder weights after low-memory loading.

    Transformers materializes missing parameters with ``torch.empty`` when
    ``low_cpu_mem_usage=True``. ChatGLM2 deliberately implements
    ``_init_weights`` as a no-op, so a newly added PrefixEncoder would otherwise
    retain uninitialized storage.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to initialize PrefixEncoder parameters") from exc

    reset_modules = []
    with torch.no_grad():
        for name, module in prefix_encoder.named_modules():
            owns_parameters = any(True for _ in module.parameters(recurse=False))
            reset_parameters = getattr(module, "reset_parameters", None)
            if owns_parameters and callable(reset_parameters):
                reset_parameters()
                reset_modules.append(name or "<root>")

    if not reset_modules:
        raise RuntimeError("PrefixEncoder exposes no resettable parameter modules")
    non_finite = [
        name
        for name, parameter in prefix_encoder.named_parameters()
        if not bool(torch.isfinite(parameter).all().detach().cpu())
    ]
    if non_finite:
        raise FloatingPointError(
            f"PrefixEncoder initialization produced non-finite parameters: {non_finite[:10]}"
        )
    return reset_modules


def stable_causal_lm_loss(logits, labels):
    """Compute shifted causal-LM loss in FP32 and reject invalid supervision.

    ChatGLM2's remote ``forward`` computes cross entropy in FP32 but casts the
    scalar back to the hidden-state dtype before returning it. Under FP16
    training a sufficiently small, still useful loss therefore underflows to
    an exact zero. Computing from the returned logits keeps that loss in FP32.
    """
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to compute training loss") from exc

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"Expected logits [batch, sequence, vocabulary] and labels [batch, sequence], "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Logit/label sequence shapes differ: {tuple(logits.shape[:2])} != "
            f"{tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("Causal-LM loss requires at least two sequence positions")

    shift_labels = labels[:, 1:].contiguous()
    supervised_mask = shift_labels.ne(-100)
    supervised_tokens = supervised_mask.sum()
    if int(supervised_tokens.detach().cpu()) == 0:
        raise ValueError("Batch has no supervised target tokens after the causal shift")

    # Select the short answer span before promoting to FP32. Casting the full
    # prompt logits would add hundreds of MB for a 65k-token vocabulary.
    supervised_logits = logits[:, :-1, :][supervised_mask].float()
    supervised_labels = shift_labels[supervised_mask]
    loss = F.cross_entropy(
        supervised_logits,
        supervised_labels,
    )
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError(
            "Non-finite causal-LM loss detected; aborting instead of logging it as 0.0"
        )
    return loss


class StableCausalLossTrainerMixin:
    """Use the stable loss and keep Trainer from rounding tiny losses to zero."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Training batch is missing labels")
        model_inputs = dict(inputs)
        model_inputs.pop("labels")
        outputs = model(**model_inputs)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, Mapping):
            logits = outputs["logits"]
        else:
            logits = outputs[0]
        loss = stable_causal_lm_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        # Trainer divides the detached step loss by gradient accumulation before
        # summing it. Restore the per-microbatch value for precise logging.
        value = float(loss.float().item()) * self.args.gradient_accumulation_steps
        self._tqpt_loss_sum = getattr(self, "_tqpt_loss_sum", 0.0) + value
        self._tqpt_loss_count = getattr(self, "_tqpt_loss_count", 0) + 1
        return loss

    def log(self, logs):
        if "loss" in logs and getattr(self, "_tqpt_loss_count", 0):
            logs = dict(logs)
            logs["loss"] = self._tqpt_loss_sum / self._tqpt_loss_count
            self._tqpt_loss_sum = 0.0
            self._tqpt_loss_count = 0
        return super().log(logs)


def resolve_eos_token_id(tokenizer) -> int:
    """Resolve EOS through ChatGLM2's native command API when v1.0 leaves it unset."""
    if hasattr(tokenizer, "get_command"):
        try:
            return int(tokenizer.get_command("<eos>"))
        except (AssertionError, KeyError, TypeError, ValueError):
            pass
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer does not expose an EOS token ID")
    return int(eos_token_id)


def build_causal_example(
    tokenizer,
    prompt: str,
    answer: str,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, list[int]]:
    """Create ChatGLM2 causal labels with the complete prompt masked by -100."""
    if hasattr(tokenizer, "build_prompt"):
        prompt = tokenizer.build_prompt(prompt, history=[])
    prompt_ids = tokenizer.encode(
        text=prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_length,
    )
    answer_ids = tokenizer.encode(
        text=answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )
    eos_token_id = resolve_eos_token_id(tokenizer)
    input_ids = list(prompt_ids) + list(answer_ids) + [eos_token_id]
    labels = [-100] * len(prompt_ids) + list(answer_ids) + [eos_token_id]
    if len(input_ids) != len(labels):
        raise AssertionError("input_ids and labels diverged")
    return {"input_ids": input_ids, "labels": labels}


def trainable_parameter_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_only_lora_trainable(model) -> list[str]:
    names = trainable_parameter_names(model)
    if not names:
        raise AssertionError("No trainable LoRA parameters found")
    unexpected = [name for name in names if "lora_" not in name.casefold()]
    if unexpected:
        raise AssertionError(f"Non-LoRA parameters are trainable: {unexpected[:10]}")
    return names


def freeze_except_prefix(model) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    try:
        prefix_encoder = model.transformer.prefix_encoder
    except AttributeError as exc:
        raise AttributeError("ChatGLM2 model does not expose transformer.prefix_encoder") from exc
    for parameter in prefix_encoder.parameters():
        parameter.requires_grad = True
    return assert_only_prefix_trainable(model)


def assert_only_prefix_trainable(model) -> list[str]:
    names = trainable_parameter_names(model)
    if not names:
        raise AssertionError("No trainable PrefixEncoder parameters found")
    unexpected = [name for name in names if "prefix_encoder" not in name]
    if unexpected:
        raise AssertionError(f"Non-prefix parameters are trainable: {unexpected[:10]}")
    return names


def prefix_state_dict(model) -> dict:
    state = model.state_dict()
    selected = {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if "prefix_encoder" in name
    }
    if not selected:
        raise ValueError("Model state contains no PrefixEncoder weights")
    return selected


def strip_prefix_state_dict(state: Mapping[str, object]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for name, value in state.items():
        marker = "transformer.prefix_encoder."
        if name.startswith(marker):
            selected[name[len(marker) :]] = value
        elif name.startswith("prefix_encoder."):
            selected[name[len("prefix_encoder.") :]] = value
    if not selected:
        raise ValueError("Checkpoint contains no recognized PrefixEncoder keys")
    return selected


def sha256_file(path: Union[Path, str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(path: Union[Path, str]) -> str:
    directory = Path(path)
    model_path = directory / "tokenizer.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"Cannot fingerprint tokenizer at {directory}")

    digest = hashlib.sha256(b"TQPT tokenizer fingerprint v2\0")
    semantic_files = (
        "tokenizer.model",
        "tokenization_chatglm.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.json",
    )
    for name in semantic_files:
        file_path = directory / name
        if not file_path.is_file():
            continue
        payload = file_path.read_bytes()
        if file_path.suffix == ".json":
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid tokenizer JSON file: {file_path}") from exc
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    report_path = directory / "traffic_tokenizer_report.json"
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        token_to_id = report.get("token_to_id")
        if token_to_id is not None:
            payload = json.dumps(
                token_to_id,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(b"report.token_to_id\0")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def select_generation_auto_class(config, causal_class, seq2seq_class):
    """Select the Transformers auto class supported by a ChatGLM snapshot.

    Early ChatGLM2 snapshots register the conditional-generation class only
    under AutoModelForSeq2SeqLM even though the model remains decoder-only.
    Later snapshots additionally register AutoModelForCausalLM.
    """
    auto_map = getattr(config, "auto_map", None) or {}
    if "AutoModelForCausalLM" in auto_map or not auto_map:
        return causal_class
    if "AutoModelForSeq2SeqLM" in auto_map:
        return seq2seq_class
    raise ValueError(
        "Model config registers neither AutoModelForCausalLM nor "
        "AutoModelForSeq2SeqLM"
    )


def write_adapter_metadata(directory: Union[Path, str], metadata: Mapping) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "tqpt_adapter.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, ensure_ascii=False, indent=2)
    return path


def validate_adapter_metadata(
    directory: Union[Path, str],
    *,
    adapter_type: str,
    task_code: Optional[str] = None,
    label_registry_sha256: Optional[str] = None,
) -> dict:
    path = Path(directory) / "tqpt_adapter.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("adapter_type") != adapter_type:
        raise ValueError(
            f"Adapter type mismatch: expected {adapter_type}, got {metadata.get('adapter_type')}"
        )
    if task_code is not None and metadata.get("task_code") != task_code:
        raise ValueError(
            f"Task mismatch: expected {task_code}, got {metadata.get('task_code')}"
        )
    if (
        label_registry_sha256 is not None
        and metadata.get("label_registry_sha256") != label_registry_sha256
    ):
        raise ValueError("Adapter label registry fingerprint does not match the active labels")
    return metadata


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def select_best_checkpoint(output_dir: Union[Path, str]) -> Path:
    root = Path(output_dir)
    state_files = sorted(root.glob("**/trainer_state.json"))
    for state_path in reversed(state_files):
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        best = state.get("best_model_checkpoint")
        if best and Path(best).is_dir():
            return Path(best)
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        match = CHECKPOINT_RE.search(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    if checkpoints:
        return max(checkpoints)[1]
    if (root / "adapter_config.json").is_file() or (root / "pytorch_model.bin").is_file():
        return root
    raise FileNotFoundError(f"No usable checkpoint found under {root}")

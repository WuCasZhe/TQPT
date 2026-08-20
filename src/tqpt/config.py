from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union


def load_config(path: Union[Path, str]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read TQPT configuration") from exc
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    required = {"project", "paths", "tokenizer", "stage1", "stage2", "evaluation", "experiments"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")
    config["_config_path"] = config_path
    config["_repo_root"] = config_path.parent.parent
    return config


def config_path(config: Mapping[str, Any], key: str) -> Path:
    value = Path(str(config["paths"][key]))
    return value if value.is_absolute() else Path(config["_repo_root"]) / value


def require_local_model_snapshot(path: Union[Path, str], *, require_weights: bool = True) -> Path:
    directory = Path(path)
    required = [
        directory / "config.json",
        directory / "tokenizer.model",
        directory / "tokenization_chatglm.py",
    ]
    if require_weights:
        required.extend([directory / "configuration_chatglm.py", directory / "modeling_chatglm.py"])
    missing = [item.name for item in required if not item.is_file()]
    weight_patterns = ("pytorch_model*.bin", "model*.safetensors")
    has_weights = any(any(directory.glob(pattern)) for pattern in weight_patterns)
    if missing or (require_weights and not has_weights):
        details = []
        if missing:
            details.append(f"missing {missing}")
        if require_weights and not has_weights:
            details.append("missing local weight shards")
        raise FileNotFoundError(
            f"Incomplete local ChatGLM2 snapshot at {directory}: {', '.join(details)}. "
            "TQPT never downloads model files automatically."
        )
    return directory

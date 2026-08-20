from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Union

from .config import config_path, load_config, require_local_model_snapshot
from .data import iter_json_records
from .registry import TASKS
from .training import sha256_file


BASE_EFFECTIVE_VOCAB_SIZE = 64794
ADDED_TOKEN_COUNT = 512
EXTENDED_EFFECTIVE_VOCAB_SIZE = 65306
MATRIX_CAPACITY = 65536
TOKENIZER_SCHEMA_VERSION = 4

CHATGLM2_UNK_TOKEN = "<unk>"
CHATGLM2_EOS_TOKEN = "</s>"
CHATGLM2_NATIVE_SPECIAL_TOKENS = ("[MASK]", "[gMASK]", "[sMASK]", "sop", "eop")

CORE_DOMAIN_TOKENS = (
    "<packet>",
    "</packet>",
    "<FLOW>",
    "</FLOW>",
    "<FIELD_SEP>",
    "<IPV4>",
    "<IPV6>",
    "<MAC>",
    "<DOMAIN>",
    "<SESSION_ID>",
    "<LONG_PAYLOAD>",
    "<EMPTY>",
    "frame.encap_type",
    "frame.time",
    "frame.time_epoch",
    "frame.time_delta",
    "frame.time_relative",
    "frame.number",
    "frame.len",
    "frame.protocols",
    "eth.src",
    "eth.dst",
    "eth.type",
    "ip.version",
    "ip.hdr_len",
    "ip.dsfield",
    "ip.dsfield.dscp",
    "ip.dsfield.ecn",
    "ip.len",
    "ip.id",
    "ip.flags",
    "ip.flags.rb",
    "ip.flags.df",
    "ip.flags.mf",
    "ip.frag_offset",
    "ip.ttl",
    "ip.proto",
    "ip.checksum",
    "ip.checksum.status",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.stream",
    "tcp.len",
    "tcp.seq",
    "tcp.nxtseq",
    "tcp.ack",
    "tcp.hdr_len",
    "tcp.flags",
    "tcp.flags.res",
    "tcp.flags.ns",
    "tcp.flags.cwr",
    "tcp.flags.ecn",
    "tcp.flags.urg",
    "tcp.flags.ack",
    "tcp.flags.push",
    "tcp.flags.reset",
    "tcp.flags.syn",
    "tcp.flags.fin",
    "tcp.window_size",
    "tcp.checksum",
    "tcp.checksum.status",
    "tcp.urgent_pointer",
    "tcp.time_relative",
    "tcp.time_delta",
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "udp.checksum",
    "udp.checksum.status",
    "udp.stream",
    "tls.record",
    "tls.handshake",
    "tls.app_data",
    "dns.id",
    "dns.flags",
    "dns.qry.name",
    "http.request.method",
    "http.host",
    "http.request.uri",
    "SYN",
    "ACK",
    "FIN",
    "RST",
    "PSH",
    "URG",
    "ECE",
    "CWR",
    "TCP",
    "UDP",
    "TLS",
    "SSL",
    "DNS",
    "HTTP",
    "DHCP",
    "ARP",
    "ICMP",
)

FIELD_RE = re.compile(r"(?:^|[,\s])([A-Za-z][A-Za-z0-9_.-]{2,})(?=\s*:)")
PROTOCOLS_RE = re.compile(r"frame\.protocols\s*:\s*([^,\n]+)", flags=re.IGNORECASE)
HEX_RE = re.compile(r"(?<![0-9A-Fa-f])0x[0-9A-Fa-f]{2,8}(?![0-9A-Fa-f])")


def _raw_dataset_files(raw_root: Path) -> Iterator[Path]:
    for spec in TASKS.values():
        directory = raw_root / spec.dataset_dir
        yield directory / spec.train_file
        yield directory / spec.test_file


def mine_domain_candidates(raw_root: Union[Path, str], max_records: Optional[int] = None) -> list[str]:
    counts: Counter[str] = Counter()
    consumed = 0
    for path in _raw_dataset_files(Path(raw_root)):
        for record in iter_json_records(path):
            text = str(record.get("instruction", ""))
            for field in FIELD_RE.findall(text):
                counts[field] += 1
            for protocol_chain in PROTOCOLS_RE.findall(text):
                for protocol in protocol_chain.split(":"):
                    protocol = protocol.strip()
                    if 2 <= len(protocol) <= 32:
                        counts[protocol] += 1
            for token in HEX_RE.findall(text):
                counts[token] += 1
            consumed += 1
            if max_records is not None and consumed >= max_records:
                return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def select_appended_tokens(
    base_tokenizer,
    candidates: Iterable[str],
    count: int = ADDED_TOKEN_COUNT,
) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")
    base_vocab = set(base_tokenizer.get_vocab())
    selected: list[str] = []
    seen = set(base_vocab)
    for token in (*CORE_DOMAIN_TOKENS, *tuple(candidates)):
        token = str(token).strip()
        if not token or token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) == count:
            return selected
    # The real corpora yield substantially more than 512 candidates. Stable
    # reserve tokens keep fixture/smoke builds exact without inventing IDs at
    # load time later.
    index = 0
    while len(selected) < count:
        token = f"<TRAFFIC_RESERVED_{index:04d}>"
        index += 1
        if token in seen:
            continue
        selected.append(token)
        seen.add(token)
    return selected


def capture_subword_initializers(base_tokenizer, tokens: Sequence[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for token in tokens:
        ids = list(base_tokenizer.encode(token, add_special_tokens=False))
        ids = [int(token_id) for token_id in ids if int(token_id) < BASE_EFFECTIVE_VOCAB_SIZE]
        result[token] = ids
    return result


def _configure_chatglm2_extension_compatibility(base_tokenizer) -> None:
    """Prepare the ChatGLM2 v1.0 slow tokenizer for Transformers 4.30.2.

    The v1.0 remote tokenizer does not register an unknown token, so the slow
    tokenizer's ``add_tokens`` implementation treats every candidate as an
    existing token.  Registering ``<unk>`` fixes that test, but is not enough:
    the remote ``get_vocab`` collapses several native special tokens to the
    empty-string key.  Some 4.30.x paths consequently choose 64790 as the first
    added ID even though IDs through 64793 are occupied.  IDs are normalized
    explicitly after insertion below, using ``vocab_size`` as the native ID
    boundary rather than the lossy dictionary length.
    """
    native_size = int(base_tokenizer.vocab_size)
    if native_size != BASE_EFFECTIVE_VOCAB_SIZE:
        raise ValueError(
            f"Expected {BASE_EFFECTIVE_VOCAB_SIZE} effective ChatGLM2 tokens, found {native_size}"
        )

    # Reading the inherited property while it is unset emits the noisy
    # "Using unk_token, but it is not set yet" warning.  Inspect its backing
    # field first; newer remote-code variants may expose a read-only property.
    if getattr(base_tokenizer, "_unk_token", None) is None:
        try:
            base_tokenizer.unk_token = CHATGLM2_UNK_TOKEN
        except AttributeError:
            pass
    configured_unk_token = base_tokenizer.unk_token
    if configured_unk_token != CHATGLM2_UNK_TOKEN:
        raise ValueError(
            f"Unexpected ChatGLM2 unknown token: {configured_unk_token!r}"
        )
    unk_id = base_tokenizer.convert_tokens_to_ids(CHATGLM2_UNK_TOKEN)
    if unk_id is None:
        raise ValueError("ChatGLM2 <unk> token does not resolve to a native ID")

    try:
        special_ids = [int(base_tokenizer.get_command(token)) for token in CHATGLM2_NATIVE_SPECIAL_TOKENS]
    except (AttributeError, AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tokenizer is missing the native ChatGLM2 command tokens") from exc
    expected_special_ids = list(
        range(BASE_EFFECTIVE_VOCAB_SIZE - len(CHATGLM2_NATIVE_SPECIAL_TOKENS), BASE_EFFECTIVE_VOCAB_SIZE)
    )
    if special_ids != expected_special_ids:
        raise ValueError(
            f"Unexpected ChatGLM2 command token IDs: {special_ids} != {expected_special_ids}"
        )


def _normalize_added_token_ids(base_tokenizer, tokens: Sequence[str]) -> dict[str, int]:
    """Move newly added entries behind every native ID, avoiding 4.30.2 collisions."""
    encoder = getattr(base_tokenizer, "added_tokens_encoder", None)
    decoder = getattr(base_tokenizer, "added_tokens_decoder", None)
    if not isinstance(encoder, dict) or not isinstance(decoder, dict):
        raise TypeError("Expected a slow Transformers tokenizer with mutable added-token maps")

    token_strings = [str(token) for token in tokens]
    unexpected = set(encoder).difference(token_strings)
    if unexpected:
        raise ValueError(f"Base tokenizer already has added tokens: {sorted(unexpected)!r}")

    mapping = {
        token: BASE_EFFECTIVE_VOCAB_SIZE + offset
        for offset, token in enumerate(token_strings)
    }
    encoder.clear()
    encoder.update(mapping)
    decoder.clear()
    decoder.update({token_id: token for token, token_id in mapping.items()})
    return mapping


def append_tokens(base_tokenizer, tokens: Sequence[str]) -> dict[str, int]:
    try:
        from transformers import AddedToken
    except ImportError as exc:
        raise RuntimeError("transformers is required to append tokenizer entries") from exc

    _configure_chatglm2_extension_compatibility(base_tokenizer)
    before_vocab = dict(base_tokenizer.get_vocab())
    if base_tokenizer.get_added_vocab():
        raise ValueError("Base ChatGLM2 tokenizer unexpectedly contains added tokens")
    added = base_tokenizer.add_tokens(
        [
            AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)
            for token in tokens
        ]
    )
    if added != len(tokens):
        raise AssertionError(f"Requested {len(tokens)} appended tokens, tokenizer added {added}")
    mapping = _normalize_added_token_ids(base_tokenizer, tokens)
    after_vocab = base_tokenizer.get_vocab()
    for token, token_id in before_vocab.items():
        if after_vocab.get(token) != token_id:
            raise AssertionError(f"Original token ID changed for {token!r}")
    converted_mapping = {token: int(base_tokenizer.convert_tokens_to_ids(token)) for token in tokens}
    if converted_mapping != mapping:
        raise AssertionError("Tokenizer did not retain the normalized appended-token mapping")
    expected = list(range(BASE_EFFECTIVE_VOCAB_SIZE, BASE_EFFECTIVE_VOCAB_SIZE + len(tokens)))
    if sorted(mapping.values()) != expected:
        raise AssertionError("Appended token IDs are not contiguous")
    for token, token_id in mapping.items():
        encoded = list(base_tokenizer.encode(token, add_special_tokens=False))
        if encoded != [token_id]:
            raise AssertionError(f"Domain token is not atomic: {token!r} -> {encoded}")
    effective_size = int(base_tokenizer.vocab_size) + len(base_tokenizer.get_added_vocab())
    if effective_size != EXTENDED_EFFECTIVE_VOCAB_SIZE:
        raise AssertionError(
            f"Expected effective vocab {EXTENDED_EFFECTIVE_VOCAB_SIZE}, found {effective_size}"
        )
    return mapping


def _get_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        try:
            input_embedding = model.get_input_embeddings()
            if input_embedding is not None:
                return input_embedding
        except (AttributeError, NotImplementedError):
            pass
    try:
        return model.transformer.embedding.word_embeddings
    except AttributeError:
        return None


def _set_input_embeddings(model, module) -> None:
    if hasattr(model, "set_input_embeddings"):
        try:
            model.set_input_embeddings(module)
            return
        except (AttributeError, NotImplementedError):
            pass
    model.transformer.embedding.word_embeddings = module


def _get_output_embeddings(model):
    if hasattr(model, "get_output_embeddings"):
        try:
            output = model.get_output_embeddings()
            if output is not None:
                return output
        except (AttributeError, NotImplementedError):
            pass
    try:
        return model.transformer.output_layer
    except AttributeError:
        return None


def _set_output_embeddings(model, module) -> None:
    if hasattr(model, "set_output_embeddings"):
        try:
            model.set_output_embeddings(module)
            return
        except (AttributeError, NotImplementedError):
            pass
    model.transformer.output_layer = module


def resize_chatglm2_embeddings(
    model,
    token_to_id: Mapping[str, int],
    subword_ids: Mapping[str, Sequence[int]],
    *,
    base_effective_size: int = BASE_EFFECTIVE_VOCAB_SIZE,
    capacity: int = MATRIX_CAPACITY,
) -> dict:
    """Explicitly grow ChatGLM2 input/output matrices and mean-initialize rows.

    ChatGLM2-6B has unused padded rows, and its remote-code generic resize path
    has historically been incomplete. This function owns both matrices,
    preserves every native row, initializes the 512 semantic rows from native
    subwords, and fills the remaining capacity rows with the native mean.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to resize model embeddings") from exc

    input_embedding = _get_input_embeddings(model)
    output_embedding = _get_output_embeddings(model)
    if input_embedding is None or output_embedding is None:
        raise ValueError("Could not locate ChatGLM2 input and output embeddings")
    input_weight = input_embedding.weight.detach()
    output_weight = output_embedding.weight.detach()
    if input_weight.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("Expected two-dimensional vocabulary matrices")
    if input_weight.shape[1] != output_weight.shape[1]:
        raise ValueError("Input embedding and LM head hidden dimensions differ")
    if input_weight.shape[0] < base_effective_size or output_weight.shape[0] < base_effective_size:
        raise ValueError("Base model matrices do not cover the native effective vocabulary")
    if max(token_to_id.values(), default=-1) >= capacity:
        raise ValueError("Appended token ID exceeds requested matrix capacity")

    input_mean = input_weight[:base_effective_size].float().mean(dim=0).to(input_weight.dtype)
    output_mean = output_weight[:base_effective_size].float().mean(dim=0).to(output_weight.dtype)
    new_input = nn.Embedding(
        capacity,
        input_weight.shape[1],
        padding_idx=getattr(input_embedding, "padding_idx", None),
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    new_output = nn.Linear(
        output_weight.shape[1],
        capacity,
        bias=getattr(output_embedding, "bias", None) is not None,
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    with torch.no_grad():
        new_input.weight.copy_(input_mean.expand_as(new_input.weight))
        new_output.weight.copy_(output_mean.expand_as(new_output.weight))
        input_rows = min(input_weight.shape[0], capacity)
        output_rows = min(output_weight.shape[0], capacity)
        new_input.weight[:input_rows].copy_(input_weight[:input_rows])
        new_output.weight[:output_rows].copy_(output_weight[:output_rows])
        if new_output.bias is not None:
            old_bias = output_embedding.bias.detach()
            bias_mean = old_bias[:base_effective_size].float().mean().to(old_bias.dtype)
            new_output.bias.fill_(bias_mean)
            new_output.bias[: min(len(old_bias), capacity)].copy_(old_bias[:capacity])
        for token, token_id in token_to_id.items():
            source_ids = [index for index in subword_ids.get(token, ()) if index < base_effective_size]
            if source_ids:
                new_input.weight[token_id].copy_(input_weight[source_ids].float().mean(dim=0).to(input_weight.dtype))
                new_output.weight[token_id].copy_(output_weight[source_ids].float().mean(dim=0).to(output_weight.dtype))

    _set_input_embeddings(model, new_input)
    _set_output_embeddings(model, new_output)
    for config in filter(None, (getattr(model, "config", None), getattr(model, "generation_config", None))):
        if hasattr(config, "vocab_size"):
            config.vocab_size = capacity
        if hasattr(config, "padded_vocab_size"):
            config.padded_vocab_size = capacity
    return {
        "old_input_rows": int(input_weight.shape[0]),
        "old_output_rows": int(output_weight.shape[0]),
        "new_input_rows": int(new_input.weight.shape[0]),
        "new_output_rows": int(new_output.weight.shape[0]),
        "hidden_size": int(new_input.weight.shape[1]),
        "capacity": capacity,
        "initialized_domain_rows": len(token_to_id),
    }


def _patch_chatglm2_tokenizer_source(source: str) -> str:
    """Persist the v1.0 compatibility fixes in copied tokenizer remote code."""
    if "self.vocab_file = vocab_file" not in source:
        v1_initializer = '    def __init__(self, vocab_file, padding_side="left", **kwargs):\n'
        if v1_initializer not in source:
            raise RuntimeError("Unsupported ChatGLM2 vocabulary initializer in tokenization_chatglm.py")
        source = source.replace(
            v1_initializer,
            f"{v1_initializer}        self.vocab_file = vocab_file\n",
            1,
        )

    vocab_old = (
        "        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}\n"
        "        vocab.update(self.added_tokens_encoder)"
    )
    vocab_new = (
        "        # TQPT: preserve one key for every native ID; v1.0 collapsed command tokens to ''.\n"
        "        vocab = {self.tokenizer.sp_model.IdToPiece(i): i for i in "
        "range(self.tokenizer.sp_model.vocab_size())}\n"
        "        vocab.update(self.tokenizer.special_tokens)\n"
        "        vocab.update(self.added_tokens_encoder)"
    )
    if vocab_old in source:
        source = source.replace(vocab_old, vocab_new, 1)
    elif "TQPT: preserve one key for every native ID" not in source:
        raise RuntimeError("Unsupported ChatGLM2 get_vocab implementation in tokenization_chatglm.py")

    pad_property = "    @property\n    def pad_token(self) -> str:"
    if not re.search(r"^\s+def eos_token\(self\)", source, flags=re.MULTILINE):
        eos_property = (
            "    # TQPT: ChatGLM2 v1.0 omitted eos_token from the Transformers API.\n"
            "    @property\n"
            "    def eos_token(self) -> str:\n"
            f"        return getattr(self, \"_eos_token\", None) or \"{CHATGLM2_EOS_TOKEN}\"\n\n"
            "    @eos_token.setter\n"
            "    def eos_token(self, value):\n"
            "        self._eos_token = value\n\n"
            f"{pad_property}"
        )
        if pad_property not in source:
            raise RuntimeError("Unsupported ChatGLM2 special-token properties in tokenization_chatglm.py")
        source = source.replace(pad_property, eos_property, 1)

    if not re.search(r"^\s+def unk_token\(self\)", source, flags=re.MULTILINE):
        unk_property = (
            "    # TQPT: ChatGLM2 v1.0 omitted unk_token, breaking add_tokens on Transformers 4.30.2.\n"
            "    @property\n"
            "    def unk_token(self) -> str:\n"
            f"        return getattr(self, \"_unk_token\", None) or \"{CHATGLM2_UNK_TOKEN}\"\n\n"
            "    @unk_token.setter\n"
            "    def unk_token(self, value):\n"
            "        self._unk_token = value\n\n"
            f"{pad_property}"
        )
        if pad_property not in source:
            raise RuntimeError("Unsupported ChatGLM2 special-token properties in tokenization_chatglm.py")
        source = source.replace(pad_property, unk_property, 1)
    return source


def ensure_chatglm2_tokenizer_remote_code(tokenizer_dir: Path) -> bool:
    """Patch an existing local ChatGLM2 tokenizer implementation in place."""
    tokenizer_source = tokenizer_dir / "tokenization_chatglm.py"
    if not tokenizer_source.is_file():
        raise FileNotFoundError(f"ChatGLM2 remote tokenizer code not found: {tokenizer_source}")
    source = tokenizer_source.read_text(encoding="utf-8")
    patched = _patch_chatglm2_tokenizer_source(source)
    if patched == source:
        return False
    tokenizer_source.write_text(patched, encoding="utf-8")
    return True


def _ensure_chatglm2_vocab_file(tokenizer, base_model: Path) -> Path:
    """Restore the vocab path omitted by ChatGLM2 v1.0's constructor."""
    configured = getattr(tokenizer, "vocab_file", None)
    vocab_file = Path(configured) if configured else base_model / "tokenizer.model"
    if not vocab_file.is_file():
        raise FileNotFoundError(f"Cannot save ChatGLM2 vocabulary; file not found: {vocab_file}")
    tokenizer.vocab_file = str(vocab_file)
    return vocab_file


def save_tokenizer_with_remote_code(tokenizer, base_model: Path, output_dir: Path) -> None:
    _ensure_chatglm2_vocab_file(tokenizer, base_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    source_code = base_model / "tokenization_chatglm.py"
    if source_code.is_file():
        destination_code = output_dir / source_code.name
        shutil.copy2(source_code, destination_code)
        ensure_chatglm2_tokenizer_remote_code(output_dir)
    config_path = output_dir / "tokenizer_config.json"
    config = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    config.update(
        {
            "tokenizer_class": "ChatGLMTokenizer",
            "auto_map": {"AutoTokenizer": ["tokenization_chatglm.ChatGLMTokenizer", None]},
            "clean_up_tokenization_spaces": False,
            "padding_side": "left",
            "unk_token": CHATGLM2_UNK_TOKEN,
            "eos_token": CHATGLM2_EOS_TOKEN,
        }
    )
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def _patch_chatglm2_prefix_cache_handling(source: str) -> str:
    """Ensure prefix KV is consumed even when output caching is disabled for training."""
    marker = "TQPT: prefix KV must be consumed independently of use_cache."
    if marker in source:
        return source
    old = (
        "        # adjust key and value for inference\n"
        "        if use_cache:\n"
        "            if kv_cache is not None:\n"
        "                cache_k, cache_v = kv_cache\n"
        "                key_layer = torch.cat((cache_k, key_layer), dim=0)\n"
        "                value_layer = torch.cat((cache_v, value_layer), dim=0)\n"
        "            kv_cache = (key_layer, value_layer)\n"
        "        else:\n"
        "            kv_cache = None"
    )
    new = (
        f"        # {marker}\n"
        "        if kv_cache is not None:\n"
        "            cache_k, cache_v = kv_cache\n"
        "            key_layer = torch.cat((cache_k, key_layer), dim=0)\n"
        "            value_layer = torch.cat((cache_v, value_layer), dim=0)\n"
        "        if use_cache:\n"
        "            kv_cache = (key_layer, value_layer)\n"
        "        else:\n"
        "            kv_cache = None"
    )
    if old in source:
        return source.replace(old, new, 1)
    # Later ChatGLM2 revisions already contain the corrected ordering.
    corrected = (
        "        if kv_cache is not None:\n"
        "            cache_k, cache_v = kv_cache\n"
        "            key_layer = torch.cat((cache_k, key_layer), dim=0)\n"
        "            value_layer = torch.cat((cache_v, value_layer), dim=0)\n"
        "        if use_cache:"
    )
    if corrected in source:
        return source
    raise RuntimeError("Unsupported ChatGLM2 KV-cache handling in modeling_chatglm.py")


def _patch_chatglm2_model_source(source: str) -> str:
    """Backport the Transformers embedding API and Prefix-Tuning to v1.0."""
    class_header = "class ChatGLMModel(ChatGLMPreTrainedModel):\n"
    if class_header not in source:
        raise RuntimeError("Unsupported ChatGLM2 model class in modeling_chatglm.py")

    embedding_marker = "TQPT: expose native embeddings to Transformers and PEFT."
    if embedding_marker not in source:
        compatibility_methods = (
            f"{class_header}"
            f"    # {embedding_marker}\n"
            "    def get_input_embeddings(self):\n"
            "        return self.embedding.word_embeddings\n\n"
            "    def set_input_embeddings(self, value):\n"
            "        self.embedding.word_embeddings = value\n\n"
            "    def get_output_embeddings(self):\n"
            "        return self.output_layer\n\n"
            "    def set_output_embeddings(self, value):\n"
            "        self.output_layer = value\n\n"
        )
        source = source.replace(class_header, compatibility_methods, 1)

    prefix_marker = "TQPT: backport native Prefix-Tuning support to ChatGLM2 v1.0."
    if prefix_marker in source or "self.prefix_encoder = PrefixEncoder(config)" in source:
        return _patch_chatglm2_prefix_cache_handling(source)

    split_function = "def split_tensor_along_last_dim"
    split_offset = source.find(split_function)
    if split_offset < 0:
        raise RuntimeError("Unsupported ChatGLM2 helper layout in modeling_chatglm.py")
    prefix_encoder = (
        "class PrefixEncoder(torch.nn.Module):\n"
        "    def __init__(self, config: ChatGLMConfig):\n"
        "        super().__init__()\n"
        "        self.prefix_projection = config.prefix_projection\n"
        "        kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2\n"
        "        self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)\n"
        "        if self.prefix_projection:\n"
        "            self.trans = torch.nn.Sequential(\n"
        "                torch.nn.Linear(kv_size, config.hidden_size),\n"
        "                torch.nn.Tanh(),\n"
        "                torch.nn.Linear(config.hidden_size, kv_size),\n"
        "            )\n\n"
        "    def forward(self, prefix: torch.Tensor):\n"
        "        prefix_tokens = self.embedding(prefix)\n"
        "        return self.trans(prefix_tokens) if self.prefix_projection else prefix_tokens\n\n\n"
    )
    source = source[:split_offset] + prefix_encoder + source[split_offset:]

    embedding_init = "        self.embedding = init_method(Embedding, config, **init_kwargs)\n"
    if embedding_init not in source:
        raise RuntimeError("Unsupported ChatGLM2 embedding initialization in modeling_chatglm.py")
    prefix_dimensions = (
        f"{embedding_init}"
        f"        # {prefix_marker}\n"
        "        self.num_layers = config.num_layers\n"
        "        self.multi_query_group_num = config.multi_query_group_num\n"
        "        self.kv_channels = config.kv_channels\n"
    )
    source = source.replace(embedding_init, prefix_dimensions, 1)

    checkpointing_init = "        self.gradient_checkpointing = False\n"
    if checkpointing_init not in source:
        raise RuntimeError("Unsupported ChatGLM2 model initialization in modeling_chatglm.py")
    prefix_init = (
        "        self.pre_seq_len = getattr(config, \"pre_seq_len\", None)\n"
        "        self.prefix_projection = getattr(config, \"prefix_projection\", False)\n"
        "        if self.pre_seq_len is not None:\n"
        "            for param in self.parameters():\n"
        "                param.requires_grad = False\n"
        "            self.prefix_tokens = torch.arange(self.pre_seq_len).long()\n"
        "            self.prefix_encoder = PrefixEncoder(config)\n"
        "            self.dropout = torch.nn.Dropout(0.1)\n"
        f"{checkpointing_init}"
    )
    source = source.replace(checkpointing_init, prefix_init, 1)

    forward_header = "    def forward(\n            self,\n            input_ids,\n"
    model_offset = source.index(class_header)
    forward_offset = source.find(forward_header, model_offset)
    if forward_offset < 0:
        raise RuntimeError("Unsupported ChatGLM2 forward method in modeling_chatglm.py")
    get_prompt = (
        "    def get_prompt(self, batch_size, device, dtype=torch.half):\n"
        "        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)\n"
        "        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)\n"
        "        past_key_values = past_key_values.view(\n"
        "            batch_size, self.pre_seq_len, self.num_layers * 2,\n"
        "            self.multi_query_group_num, self.kv_channels,\n"
        "        )\n"
        "        past_key_values = self.dropout(past_key_values)\n"
        "        return past_key_values.permute([2, 1, 0, 3, 4]).split(2)\n\n"
    )
    source = source[:forward_offset] + get_prompt + source[forward_offset:]

    input_embedding = (
        "        if inputs_embeds is None:\n"
        "            inputs_embeds = self.embedding(input_ids)\n\n"
    )
    if input_embedding not in source:
        raise RuntimeError("Unsupported ChatGLM2 forward embedding block in modeling_chatglm.py")
    prefix_forward = (
        f"{input_embedding}"
        "        if self.pre_seq_len is not None:\n"
        "            if past_key_values is None:\n"
        "                past_key_values = self.get_prompt(\n"
        "                    batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype\n"
        "                )\n"
        "            if attention_mask is not None:\n"
        "                prefix_mask = attention_mask.new_ones((batch_size, self.pre_seq_len))\n"
        "                attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)\n\n"
    )
    source = source.replace(input_embedding, prefix_forward, 1)
    return _patch_chatglm2_prefix_cache_handling(source)


def ensure_chatglm2_model_remote_code(model_dir: Path) -> bool:
    """Patch an existing local ChatGLM2 artifact in place when required."""
    model_source = model_dir / "modeling_chatglm.py"
    if not model_source.is_file():
        raise FileNotFoundError(f"ChatGLM2 remote model code not found: {model_source}")
    source = model_source.read_text(encoding="utf-8")
    patched = _patch_chatglm2_model_source(source)
    if patched == source:
        return False
    model_source.write_text(patched, encoding="utf-8")
    return True


def copy_model_remote_code(base_model: Path, output_dir: Path) -> None:
    for name in (
        "configuration_chatglm.py",
        "modeling_chatglm.py",
        "quantization.py",
        "generation_config.json",
    ):
        source = base_model / name
        destination = output_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    ensure_chatglm2_model_remote_code(output_dir)


def _effective_tokenizer_size(tokenizer) -> int:
    return int(tokenizer.vocab_size) + len(tokenizer.get_added_vocab())


def verify_reloaded_tokenizer(base_tokenizer, extended_tokenizer, token_to_id: Mapping[str, int]) -> dict:
    base_vocab = base_tokenizer.get_vocab()
    extended_vocab = extended_tokenizer.get_vocab()
    if int(base_tokenizer.vocab_size) != BASE_EFFECTIVE_VOCAB_SIZE:
        raise AssertionError("Unexpected native tokenizer size during reload verification")
    if _effective_tokenizer_size(extended_tokenizer) != EXTENDED_EFFECTIVE_VOCAB_SIZE:
        raise AssertionError("Saved tokenizer did not reload with the expected effective size")
    if getattr(extended_tokenizer, "unk_token", None) != CHATGLM2_UNK_TOKEN:
        raise AssertionError("Saved tokenizer lost the ChatGLM2 unknown-token compatibility setting")
    for token in CHATGLM2_NATIVE_SPECIAL_TOKENS:
        if extended_tokenizer.get_command(token) != base_tokenizer.get_command(token):
            raise AssertionError(f"Native command token ID changed after reload: {token!r}")
    for token, token_id in base_vocab.items():
        # ChatGLM2 v1.0's native get_vocab aliases several command IDs to an
        # empty key.  The persisted compatibility patch restores their names,
        # so that lossy placeholder is deliberately excluded from comparison.
        if token and extended_vocab.get(token) != token_id:
            raise AssertionError(f"Native token ID changed after reload: {token!r}")
    for token, token_id in token_to_id.items():
        if extended_tokenizer.get_added_vocab().get(token) != token_id:
            raise AssertionError(f"Appended token ID changed after reload: {token!r}")
        if list(extended_tokenizer.encode(token, add_special_tokens=False)) != [token_id]:
            raise AssertionError(f"Appended token lost atomic encoding after reload: {token!r}")
    return {
        "base_vocab_size": int(base_tokenizer.vocab_size),
        "extended_vocab_size": _effective_tokenizer_size(extended_tokenizer),
        "first_added_token_id": min(token_to_id.values()),
        "last_added_token_id": max(token_to_id.values()),
        "atomic_tokens": len(token_to_id),
    }


def compression_metrics(base_tokenizer, extended_tokenizer, raw_root: Path, sample_limit: int) -> dict:
    samples = 0
    base_count = 0
    extended_count = 0
    max_extended_id = -1
    for path in _raw_dataset_files(raw_root):
        for record in iter_json_records(path):
            text = str(record.get("instruction", ""))
            base_ids = list(base_tokenizer.encode(text, add_special_tokens=False))
            extended_ids = list(extended_tokenizer.encode(text, add_special_tokens=False))
            base_count += len(base_ids)
            extended_count += len(extended_ids)
            if extended_ids:
                max_extended_id = max(max_extended_id, max(extended_ids))
            samples += 1
            if samples >= sample_limit:
                if max_extended_id >= MATRIX_CAPACITY:
                    raise AssertionError("Tokenizer emitted an ID outside model matrix capacity")
                return {
                    "samples": samples,
                    "base_token_count": base_count,
                    "extended_token_count": extended_count,
                    "compression_ratio": extended_count / base_count if base_count else None,
                    "max_extended_token_id": max_extended_id,
                }
    return {
        "samples": samples,
        "base_token_count": base_count,
        "extended_token_count": extended_count,
        "compression_ratio": extended_count / base_count if base_count else None,
        "max_extended_token_id": max_extended_id,
    }


def build_tokenizer_and_model(
    base_model: Path,
    raw_root: Path,
    tokenizer_output: Path,
    model_output: Optional[Path],
    *,
    max_records: Optional[int] = None,
    compression_samples: int = 1000,
) -> dict:
    require_local_model_snapshot(base_model, require_weights=model_output is not None)
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the locked training dependencies before building the tokenizer") from exc

    native_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    candidates = mine_domain_candidates(raw_root, max_records=max_records)
    tokens = select_appended_tokens(base_tokenizer, candidates)
    initializers = capture_subword_initializers(base_tokenizer, tokens)
    token_to_id = append_tokens(base_tokenizer, tokens)
    save_tokenizer_with_remote_code(base_tokenizer, base_model, tokenizer_output)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_output, trust_remote_code=True)
    reload_report = verify_reloaded_tokenizer(native_tokenizer, reloaded_tokenizer, token_to_id)
    compression = compression_metrics(
        native_tokenizer,
        reloaded_tokenizer,
        raw_root,
        sample_limit=compression_samples,
    )

    base_model_file = base_model / "tokenizer.model"
    saved_model_file = tokenizer_output / "tokenizer.model"
    base_hash = sha256_file(base_model_file)
    saved_hash = sha256_file(saved_model_file)
    if base_hash != saved_hash:
        raise AssertionError("The native SentencePiece model changed during append-only extension")

    resize_report = None
    if model_output is not None:
        import torch

        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        resize_report = resize_chatglm2_embeddings(model, token_to_id, initializers)
        model_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_output)
        copy_model_remote_code(base_model, model_output)
        save_tokenizer_with_remote_code(base_tokenizer, base_model, model_output)

    report = {
        "schema_version": TOKENIZER_SCHEMA_VERSION,
        "base_model": str(base_model.resolve()),
        "raw_root": str(raw_root.resolve()),
        "base_vocab_size": BASE_EFFECTIVE_VOCAB_SIZE,
        "added_tokens": ADDED_TOKEN_COUNT,
        "extended_vocab_size": EXTENDED_EFFECTIVE_VOCAB_SIZE,
        "matrix_capacity": MATRIX_CAPACITY,
        "first_added_token_id": min(token_to_id.values()),
        "last_added_token_id": max(token_to_id.values()),
        "tokenizer_model_sha256": base_hash,
        "token_to_id": token_to_id,
        "new_tokens": tokens,
        "reload_verification": reload_report,
        "compression": compression,
        "resize": resize_report,
    }
    with (tokenizer_output / "traffic_tokenizer_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if model_output is not None:
        shutil.copy2(
            tokenizer_output / "traffic_tokenizer_report.json",
            model_output / "traffic_tokenizer_report.json",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the append-only ChatGLM2 traffic tokenizer")
    parser.add_argument("--config", type=Path, default=Path("configs/tqpt.yaml"))
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--tokenizer-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--max-records", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    contract = config["tokenizer"]
    expected_contract = {
        "base_effective_vocab_size": BASE_EFFECTIVE_VOCAB_SIZE,
        "added_tokens": ADDED_TOKEN_COUNT,
        "effective_vocab_size": EXTENDED_EFFECTIVE_VOCAB_SIZE,
        "matrix_capacity": MATRIX_CAPACITY,
    }
    actual_contract = {key: int(contract[key]) for key in expected_contract}
    if actual_contract != expected_contract:
        raise ValueError(
            f"Tokenizer config violates the ChatGLM2 contract: {actual_contract} != {expected_contract}"
        )
    args.base_model = args.base_model or config_path(config, "base_model")
    args.raw_root = args.raw_root or config_path(config, "raw_data")
    args.tokenizer_output = args.tokenizer_output or config_path(config, "extended_tokenizer")
    args.model_output = args.model_output or config_path(config, "extended_model")
    report = build_tokenizer_and_model(
        args.base_model,
        args.raw_root,
        args.tokenizer_output,
        None if args.tokenizer_only else args.model_output,
        max_records=args.max_records,
        compression_samples=int(contract["compression_samples"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

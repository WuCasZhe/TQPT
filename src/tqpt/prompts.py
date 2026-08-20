from __future__ import annotations

import ipaddress
import re
from typing import Optional, Tuple


PACKET_BOUNDARY = re.compile(
    r"\n\s*(?:<packet>\s*:\s*)?(?=(?:frame|eth|ip|ipv6|tcp|udp|tls|ssl|dns|http|dhcp|arp|raw)\.)",
    flags=re.IGNORECASE,
)
MAC_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])")
IPV4_RE = re.compile(
    r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
)
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
DOMAIN_RE = re.compile(
    r"(?<![\w.-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:com|net|org|edu|gov|cn|io|co|tv|me)(?![\w.-])",
    flags=re.IGNORECASE,
)
SESSION_RE = re.compile(
    r"((?:tcp|udp)\.stream\s*:\s*|session(?:_?id)?\s*[:=]\s*)[^,\s]+",
    flags=re.IGNORECASE,
)
PAYLOAD_FIELD_RE = re.compile(
    r"((?:[A-Za-z0-9_.-]*payload[A-Za-z0-9_.-]*)\s*:\s*)([^,\n]+)",
    flags=re.IGNORECASE,
)

TRAILING_SENTENCE_PUNCTUATION = ".。!！?？;；"


def split_instruction_and_traffic(text: str) -> Tuple[str, str]:
    """Split the natural-language directive from the structured packet body.

    The first literal ``<packet>`` often occurs inside the English directive,
    so splitting on that token is incorrect for the ISCXVPN records. The true
    boundary is a newline followed by an optional marker and a protocol field.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input prompt is empty")
    match = PACKET_BOUNDARY.search(text)
    if match is None:
        raise ValueError("Could not locate a structured traffic boundary")
    instruction = text[: match.start()].strip()
    traffic = text[match.end() :].strip()
    if not instruction or not traffic:
        raise ValueError("Both task instruction and traffic body must be non-empty")
    return instruction, traffic


def _normalize_payload(match: re.Match[str]) -> str:
    prefix, value = match.groups()
    compact = re.sub(r"[\s:]", "", value)
    if len(compact) >= 64:
        return prefix + "<LONG_PAYLOAD>"
    return match.group(0)


def _normalize_ipv6(match: re.Match[str]) -> str:
    candidate = match.group(0)
    try:
        ipaddress.IPv6Address(candidate)
    except ipaddress.AddressValueError:
        return candidate
    return "<IPV6>"


def normalize_traffic(traffic: str) -> str:
    """Normalize high-cardinality identifiers while retaining protocol fields."""
    value = PAYLOAD_FIELD_RE.sub(_normalize_payload, traffic)
    value = SESSION_RE.sub(lambda match: match.group(1) + "<SESSION_ID>", value)
    value = MAC_RE.sub("<MAC>", value)
    value = IPV4_RE.sub("<IPV4>", value)
    value = IPV6_RE.sub(_normalize_ipv6, value)
    value = DOMAIN_RE.sub("<DOMAIN>", value)
    return value


def build_stage2_prompt(task_instruction: str, traffic: str, *, normalize: bool = True) -> str:
    body = normalize_traffic(traffic) if normalize else traffic
    return f"{task_instruction.strip()}\n<packet>\n{body.strip()}\n</packet>"


def normalize_generated_label(text: str) -> str:
    """Apply only the output cleanup allowed by the evaluation protocol."""
    if text is None:
        return ""
    return str(text).strip().rstrip(TRAILING_SENTENCE_PUNCTUATION).rstrip()


def parse_route_code(text: str) -> Optional[str]:
    candidate = normalize_generated_label(text).upper()
    return candidate if candidate in {"EVD", "AAD", "CD"} else None

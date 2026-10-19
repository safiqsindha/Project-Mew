"""BFCL task: frozen items, prompt construction, Qwen3.5 tool-call parsing (PREREGISTRATION.md §6.2).

Qwen3.5 emits tool calls in the qwen3_coder XML format:
    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>
bfcl-eval 2026.3.23 has no Qwen3.5 handler (its QwenFCHandler parses Qwen3's JSON format), so this
module decodes the XML into bfcl's AST form [{name: {arg: value}}]. Values are typed from the tool's
JSON schema the same way vLLM's qwen3_coder tool parser does. Scoring itself uses bfcl's ast_checker
(see mew/score.py), never anything here.
"""
import ast
import json
from pathlib import Path
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def load_items(path: Path, scored_only: bool = False) -> list[dict]:
    items = [json.loads(line) for line in open(path)]
    for it in items:
        it["uid"] = it["id"]
    return [it for it in items if it["scored"]] if scored_only else items


def build_prompt(tokenizer, item: dict) -> str:
    return tokenizer.apply_chat_template(item["messages"], tools=item["tools"] or None, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def _convert(raw: str, schema: dict):
    # Same rules as vLLM's Qwen3CoderToolParser: strip one leading/trailing newline, then cast by type.
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    typ = (schema or {}).get("type", "string")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "string")
    if raw.strip().lower() == "null":
        return None
    if typ == "string":
        return raw
    if typ == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    if typ == "number":
        try:
            f = float(raw)
            return int(f) if f.is_integer() else f
        except ValueError:
            return raw
    if typ == "boolean":
        low = raw.strip().lower()
        return low == "true" if low in ("true", "false") else raw
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except Exception:
            pass
    return raw


def decode_tool_calls(completion: str, tools: list) -> list:
    """Return [{name: {arg: value}}, ...]. Raises ValueError if no well-formed call is present."""
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools}
    calls = []
    for block in TOOL_CALL_RE.findall(completion):
        for name, body in FUNCTION_RE.findall(block):
            name = name.strip()
            props = schemas.get(name, {})
            args = {p.strip(): _convert(v, props.get(p.strip())) for p, v in PARAM_RE.findall(body)}
            calls.append({name: args})
    if not calls:
        raise ValueError("no well-formed tool call")
    return calls

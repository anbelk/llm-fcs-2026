"""ДЗ5. RL для вызова инструментов."""

from pathlib import Path
import sys
import subprocess
import csv
from statistics import mean
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import platform
import re
import urllib.request
from collections import Counter
import math
import time

BASE = "Qwen/Qwen3.5-2B"
ROOT = Path(__file__).resolve().parent
REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
URL = f"https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/{REVISION}/data.json"
SOURCE_SHA256 = "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e"


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def serialized(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def norm(s):
    return " ".join(s.casefold().split())


def parts(s, separator=","):
    out, stack, quote, escaped, start = ([], [], None, False, 0)
    pairs = {")": "(", "]": "[", "}": "{"}
    for i, c in enumerate(s):
        if quote:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]:
                raise ValueError("unbalanced delimiters")
        elif c == separator and (not stack):
            out.append(s[start:i].strip())
            start = i + 1
    if quote or stack:
        raise ValueError("unclosed literal")
    out.append(s[start:].strip())
    return out


def literal(s):
    try:
        x = json.loads(s)
        json.dumps(x, allow_nan=False)
        return x
    except (ValueError, TypeError):
        x = ast.literal_eval(s)

        def check(v):
            if type(v) in (str, int, float, bool, type(None)):
                return
            if isinstance(v, list):
                for e in v:
                    check(e)
            elif isinstance(v, dict) and all((isinstance(k, str) for k in v)):
                for e in v.values():
                    check(e)
            else:
                raise ValueError("non-JSON literal")

        check(x)
        json.dumps(x, allow_nan=False)
        return x


def parse_calls(text, names):
    text = text.strip()
    if not text.startswith("["):
        return None
    if not text.endswith("]"):
        raise ValueError("bracket prefix without closing bracket")
    if not text[1:-1].strip():
        return []
    calls, body_text, pos = ([], text[1:-1], 0)
    while pos < len(body_text):
        chunk = body_text[pos:].lstrip()
        candidates = [n for n in names if chunk.startswith(n + "(")]
        if not candidates:
            raise ValueError("unknown function or non-call bracket text")
        name = max(candidates, key=len)
        start, stack, quote, escaped, end = (len(name), 0, None, False, None)
        for i in range(start, len(chunk)):
            c = chunk[i]
            if quote:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == quote:
                    quote = None
            elif c in "\"'":
                quote = c
            elif c == "(":
                stack += 1
            elif c == ")":
                stack -= 1
                if stack == 0:
                    end = i
                    break
        if end is None:
            raise ValueError("call missing closing parenthesis")
        body = chunk[len(name) + 1 : end].strip()
        args = {}
        for arg in parts(body) if body else []:
            kv = parts(arg, "=")
            if len(kv) != 2 or not kv[0] or kv[0] in args:
                raise ValueError("bad or duplicate argument")
            args[kv[0]] = literal(kv[1])
        calls.append({"name": name, "arguments": args})
        rest = chunk[end + 1 :].strip()
        if not rest:
            break
        if not rest.startswith(",") or not rest[1:].strip():
            raise ValueError("missing call separator or trailing comma")
        body_text, pos = (rest[1:].lstrip(), 0)
    return calls


def convert_schema(x):
    if isinstance(x, list):
        return [convert_schema(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {
        k: convert_schema(v)
        for k, v in x.items()
        if not (k == "required" and v is None)
    }
    if isinstance(out.get("type"), str):
        out["type"] = {
            "dict": "object",
            "int": "integer",
            "float": "number",
            "tuple": "array",
            "any": "string",
        }.get(out["type"], out["type"])
    return out


def type_ok(x, t):
    checks = {
        "string": lambda: isinstance(x, str),
        "integer": lambda: type(x) is int,
        "int": lambda: type(x) is int,
        "number": lambda: type(x) in (int, float),
        "float": lambda: type(x) in (int, float),
        "boolean": lambda: type(x) is bool,
        "dict": lambda: isinstance(x, dict),
        "object": lambda: isinstance(x, dict),
        "array": lambda: isinstance(x, list),
        "tuple": lambda: isinstance(x, list),
        "null": lambda: x is None,
        "any": lambda: True,
    }
    if isinstance(t, list):
        return any((type_ok(x, v) for v in t))
    return checks.get(t, lambda: True)()


def validate(x, schema, path=""):
    issues = []
    if not type_ok(x, schema.get("type")):
        return [("type_mismatch", path)]
    if "enum" in schema and x not in schema["enum"]:
        issues.append(("enum_mismatch", path))
    if isinstance(x, dict):
        props = schema.get("properties", {})
        for k in schema.get("required") or []:
            if k not in x:
                issues.append(("missing_required", path + "." + k))
        for k, v in x.items():
            if k not in props and (
                "properties" in schema or schema.get("additionalProperties") is False
            ):
                issues.append(("undeclared_argument", path + "." + k))
            elif k in props:
                issues.extend(validate(v, props[k], path + "." + k))
    if isinstance(x, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(x):
            issues.extend(validate(v, schema["items"], path + f"[{i}]"))
    return issues


def dist(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "n": len(values),
        "min": values[0],
        "p50": values[int((len(values) - 1) * 0.5)],
        "p90": values[int((len(values) - 1) * 0.9)],
        "p95": values[int((len(values) - 1) * 0.95)],
        "p99": values[int((len(values) - 1) * 0.99)],
        "max": values[-1],
        "mean": round(sum(values) / len(values), 2),
    }


def prepare_data(data_file, output, tokenizer_name=BASE, benchmark=None):
    output.mkdir(parents=True, exist_ok=True)
    if not data_file.exists():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, data_file)
    raw = data_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("ToolACE hash differs from the audited revision")
    data = json.loads(raw)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=MODEL_REVISION)
    counts, roles, tools_hist, user_hist, call_hist = (Counter() for _ in range(5))
    types, value_types, issues_count, seqs, argument_hist, unsupported_formats = (
        Counter() for _ in range(6)
    )
    source_chars, source_tokens, prompt_tokens, target_tokens = ([], [], [], [])
    candidates, exceptions, all_users, full_hashes, tool_sets = ([], [], [], [], [])
    schema_hashes, function_names = (set(), set())
    for di, row in enumerate(data):
        full_hashes.append(digest(row))
        conv = row["conversations"]
        source = row["system"] + "\n" + "\n".join((m["value"] for m in conv))
        source_chars.append(len(source))
        source_tokens.append(len(tokenizer.encode(source)))
        roles.update((m["from"] for m in conv))
        counts["all_assistant_turns"] += sum((m["from"] == "assistant" for m in conv))
        counts["all_decisions_after_user"] += sum(
            (
                m["from"] == "assistant" and j > 0 and (conv[j - 1]["from"] == "user")
                for j, m in enumerate(conv)
            )
        )
        seqs["/".join((m["from"] for m in conv))] += 1
        user_hist[sum((m["from"] == "user" for m in conv))] += 1
        users = [norm(m["value"]) for m in conv if m["from"] == "user"]
        all_users.append(users)
        try:
            markers = [
                "Here is a list of functions in JSON format that you can invoke:",
                "以下是可以调用的工具的JSON格式列表:",
            ]
            marker = next((s for s in markers if s in row["system"]), None)
            if marker is None:
                raise ValueError(
                    "unsupported schema/call representation; not asserted to be corrupt"
                )
            pos = row["system"].index("[", row["system"].index(marker))
            tools, _ = json.JSONDecoder().raw_decode(row["system"][pos:])
            assert isinstance(tools, list) and all(
                ("name" in t and "parameters" in t for t in tools)
            )
        except Exception as e:
            counts["schema_parse_fail_dialogues"] += 1
            for label, pattern in {
                "html": "<table",
                "xml": "<tool_name>",
                "latex": "\\begin{tabular}",
                "markdown": "**tool_name**",
                "json_object": '{"tool_name"',
                "yaml_or_plain": "tool_name:",
            }.items():
                unsupported_formats[label] += pattern in row["system"]
            exceptions.append(
                {"dialogue": di, "reason": "schema_parse", "detail": str(e)}
            )
            tool_sets.append(set())
            continue
        by_name = {t["name"]: t for t in tools}
        tools_hist[len(tools)] += 1
        tool_sets.append(set(by_name))
        function_names.update(by_name)
        for t in tools:
            schema_hashes.add(digest(t))
            types.update(
                (
                    str(p.get("type"))
                    for p in t["parameters"].get("properties", {}).values()
                )
            )
            if not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", t["name"]):
                counts["non_python_name_occurrences"] += 1
        if len(by_name) != len(tools):
            counts["duplicate_names_in_schema_dialogues"] += 1
        api_tools = [{"type": "function", "function": convert_schema(t)} for t in tools]
        prefix = row["system"].split(marker)[0]
        time_context = "\n".join(
            (
                line.strip()
                for line in prefix.splitlines()
                if re.search("\\b(today|current time|current date)\\b", line, re.I)
            )
        )
        chinese_time = re.search("今天是[^。\\n]*。", prefix)
        if chinese_time:
            time_context += chinese_time.group()
        counts["dialogues_with_time_context"] += bool(time_context)
        history = [{"role": "system", "content": time_context}] if time_context else []
        history_bad = False
        for ti, m in enumerate(conv):
            role, text = (m["from"], m["value"])
            if role != "assistant":
                history.append({"role": role, "content": text})
                continue
            counts["assistant_turns"] += 1
            after_user = ti > 0 and conv[ti - 1]["from"] == "user"
            counts["after_user" if after_user else "not_after_user"] += 1
            err, call_issues = (None, [])
            try:
                calls = parse_calls(text, by_name)
                if (
                    calls is None
                    and ti + 1 < len(conv)
                    and (conv[ti + 1]["from"] == "tool")
                ):
                    raise ValueError("tool response follows non-call answer")
                for c in calls or []:
                    call_issues.extend(
                        validate(c["arguments"], by_name[c["name"]]["parameters"])
                    )
                    counts["calls"] += 1
                    counts["arguments"] += len(c["arguments"])
                    argument_hist[len(c["arguments"])] += 1
                    value_types.update(
                        (type(v).__name__ for v in c["arguments"].values())
                    )
                    for k, p in (
                        by_name[c["name"]]["parameters"].get("properties", {}).items()
                    ):
                        if k not in c["arguments"] and k not in (
                            by_name[c["name"]]["parameters"].get("required") or []
                        ):
                            counts["omitted_optional_arguments"] += 1
            except (ValueError, SyntaxError, TypeError, KeyError, AssertionError) as e:
                calls, err = (None, str(e))
                counts["call_parse_fail_turns"] += 1
            if err or call_issues:
                exceptions.append(
                    {
                        "dialogue": di,
                        "turn": ti,
                        "reason": "call_parse" if err else "schema_validation",
                        "detail": err or call_issues,
                        "source": text,
                    }
                )
            issues_count.update((k for k, _ in call_issues))
            counts["turns_with_schema_issues"] += bool(call_issues)
            if not err:
                call_hist[len(calls or [])] += 1
                if calls and len({c["name"] for c in calls}) < len(calls):
                    counts["turns_with_repeated_function_name"] += 1
                counts["calls_turns" if calls else "text_or_empty_turns"] += 1
                if after_user:
                    counts[
                        "decision_call_turns" if calls else "decision_no_call_turns"
                    ] += 1
                else:
                    counts[
                        "post_tool_call_turns" if calls else "post_tool_no_call_turns"
                    ] += 1
            if after_user:
                reason = (
                    "bad_current_target"
                    if err or call_issues
                    else (
                        "bad_history"
                        if history_bad
                        else (
                            "duplicate_schema_names"
                            if len(by_name) != len(tools)
                            else None
                        )
                    )
                )
                if reason:
                    counts["excluded_" + reason] += 1
                else:
                    messages = list(history)
                    tokens = tokenizer.apply_chat_template(
                        messages,
                        tools=api_tools,
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    p_len = len(tokens)
                    target_message = {
                        "role": "assistant",
                        "content": text if not calls else "",
                        **(
                            {
                                "tool_calls": [
                                    {"type": "function", "function": c} for c in calls
                                ]
                            }
                            if calls
                            else {}
                        ),
                    }
                    completed = tokenizer.apply_chat_template(
                        messages + [target_message],
                        tools=api_tools,
                        tokenize=True,
                        return_dict=False,
                        enable_thinking=False,
                    )
                    assert (
                        completed[:p_len] == tokens
                    ), "prompt is not prefix of completed template"
                    t_len = len(completed) - p_len
                    prompt_tokens.append(p_len)
                    target_tokens.append(t_len)
                    ref = {
                        "kind": "calls" if calls else "no_call",
                        "calls": calls or [],
                    }
                    candidates.append(
                        {
                            "dialogue": di,
                            "turn": ti,
                            "prompt": messages,
                            "tools": api_tools,
                            "reference": ref,
                            "prompt_tokens": p_len,
                            "target_tokens": t_len,
                            "prompt_hash": digest([api_tools, messages]),
                            "reference_hash": digest(ref),
                            "fits_4096_with_1024_reserve": p_len <= 3072,
                            "reference_fits_1024": t_len <= 1024,
                        }
                    )
            history_bad |= bool(err or call_issues)
            if calls and (not err):
                history.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": c} for c in calls
                        ],
                    }
                )
            else:
                history.append({"role": "assistant", "content": text})
    parent = list(range(len(data)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = (root(a), root(b))
        parent[max(a, b)] = min(a, b)

    seen = {}
    for i, h in enumerate(full_hashes):
        if h in seen:
            union(i, seen[h])
        else:
            seen[h] = i
    prompt_sources, prompt_refs = ({}, defaultdict(set))
    for c in candidates:
        h = c["prompt_hash"]
        prompt_refs[h].add(c["reference_hash"])
        if h in prompt_sources:
            union(c["dialogue"], prompt_sources[h])
        else:
            prompt_sources[h] = c["dialogue"]
    groups = defaultdict(list)
    for i in range(len(data)):
        groups[root(i)].append(i)
    split = {}
    for ids in groups.values():
        group_key = min((full_hashes[i] for i in ids))
        name = "validation" if int(digest(["42", group_key]), 16) % 10 == 0 else "train"
        for i in ids:
            split[i] = name
    selected, dedup_seen = ([], set())
    for c in candidates:
        if len(prompt_refs[c["prompt_hash"]]) > 1:
            counts["excluded_conflicting_prompt_targets"] += 1
            continue
        if c["prompt_hash"] in dedup_seen:
            counts["excluded_duplicate_prompts"] += 1
            continue
        dedup_seen.add(c["prompt_hash"])
        c["split"] = split[c["dialogue"]]
        if not c["fits_4096_with_1024_reserve"] or not c["reference_fits_1024"]:
            counts["excluded_length"] += 1
            continue
        selected.append(c)
    user_sets = {
        s: {u for i, us in enumerate(all_users) if split[i] == s for u in us}
        for s in ("train", "validation")
    }
    names_sets = {
        s: set().union(*(ts for i, ts in enumerate(tool_sets) if split[i] == s))
        for s in ("train", "validation")
    }
    leak = {
        "shared_normalized_user_utterances": len(
            user_sets["train"] & user_sets["validation"]
        ),
        "shared_function_names": len(names_sets["train"] & names_sets["validation"]),
    }
    if benchmark:
        bfcl = [json.loads(s) for s in benchmark.read_text().splitlines() if s.strip()]
        bu = {
            norm(m["content"])
            for b in bfcl
            for m in b["question"]
            if m["role"] == "user"
        }
        bn = {t["name"] for b in bfcl for t in b["function"]}
        leak.update(
            bfcl_rows=len(bfcl),
            bfcl_sha256=hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            bfcl_exact_normalized_user_overlap=len(
                bu & set().union(*user_sets.values())
            ),
            bfcl_function_name_overlap=len(bn & function_names),
        )
    for s in ("train", "validation"):
        rows = [c for c in selected if c["split"] == s]
        (output / f"{s}.jsonl").write_text(
            "".join((serialized(c) + "\n" for c in rows))
        )
    (output / "candidates.jsonl").write_text(
        "".join((serialized(c) + "\n" for c in candidates))
    )
    (output / "exceptions.jsonl").write_text(
        "".join((serialized(c) + "\n" for c in exceptions))
    )
    (output / "split_manifest.json").write_text(
        canonical(
            {"seed": 42, "dialogue_split": split, "groups": list(groups.values())}
        )
        + "\n"
    )
    counts.update(
        dialogues=len(data),
        unique_full_dialogues=len(set(full_hashes)),
        unique_function_names=len(function_names),
        unique_function_schemas=len(schema_hashes),
        candidate_decisions=len(candidates),
        selected_decisions=len(selected),
    )
    report = {
        "source": {
            "url": URL,
            "revision": REVISION,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("transformers", "tokenizers", "jinja2")},
        },
        "model_revision": MODEL_REVISION,
        "counts": counts,
        "roles": roles,
        "role_sequences": seqs,
        "available_tools_per_dialogue": tools_hist,
        "user_turns_per_dialogue": user_hist,
        "calls_per_assistant_turn": call_hist,
        "declared_parameter_types_occurrences": types,
        "arguments_per_call": argument_hist,
        "unsupported_schema_format_markers_nonexclusive": unsupported_formats,
        "observed_argument_value_types": value_types,
        "schema_issues": {
            k: issues_count[k]
            for k in (
                "undeclared_argument",
                "missing_required",
                "type_mismatch",
                "enum_mismatch",
            )
        },
        "lengths": {
            "source_dialogue_chars": dist(source_chars),
            "source_dialogue_tokens_no_template": dist(source_tokens),
            "candidate_prompt_tokens": dist(prompt_tokens),
            "candidate_target_tokens": dist(target_tokens),
            "candidate_prompt_gt3072": sum((n > 3072 for n in prompt_tokens)),
            "candidate_prompt_gt4096": sum((n > 4096 for n in prompt_tokens)),
            "candidate_reference_gt1024": sum((n > 1024 for n in target_tokens)),
        },
        "duplicates": {
            "full_dialogue_extra_copies": len(data) - len(set(full_hashes)),
            "candidate_prompt_extra_copies": len(candidates) - len(prompt_refs),
            "conflicting_prompt_groups": sum(
                (len(v) > 1 for v in prompt_refs.values())
            ),
        },
        "split": {
            s: {
                "dialogues": sum((v == s for v in split.values())),
                "dialogues_contributing_selected": len(
                    {c["dialogue"] for c in selected if c["split"] == s}
                ),
                "selected_decisions": sum((c["split"] == s for c in selected)),
                "selected_no_call": sum(
                    (
                        c["split"] == s and c["reference"]["kind"] == "no_call"
                        for c in selected
                    )
                ),
            }
            for s in ("train", "validation")
        },
        "overlap": leak,
    }
    (output / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def typed_key(value, numeric_equivalence=False):
    if numeric_equivalence and type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return ("number", value)
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    ((k, typed_key(v, numeric_equivalence)) for k, v in value.items())
                )
            ),
        )
    if isinstance(value, list):
        return ("list", tuple((typed_key(v, numeric_equivalence) for v in value)))
    if type(value) is float and (not math.isfinite(value)):
        raise ValueError("non-finite number")
    return (type(value).__name__, value)


def call_key(call, numeric_equivalence=False):
    return (call["name"], typed_key(call["arguments"], numeric_equivalence))


def pair_score(pred, gold, numeric_equivalence=False):
    if pred["name"] != gold["name"]:
        return 0.0
    a, b = (pred["arguments"], gold["arguments"])
    common = sum(
        (
            k in b
            and typed_key(v, numeric_equivalence)
            == typed_key(b[k], numeric_equivalence)
            for k, v in a.items()
        )
    )
    f = 2 * common / (len(a) + len(b)) if a or b else 1.0
    return 0.25 + 0.75 * f


def maximum_assignment(weights):
    n = max(len(weights), len(weights[0]) if weights else 0)
    if not n:
        return 0.0
    cost = [
        [
            -(weights[i][j] if i < len(weights) and j < len(weights[i]) else 0.0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    u, v, p, way = ([0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1))
    for i in range(1, n + 1):
        p[0], j0 = (i, 0)
        minv, used = ([math.inf] * (n + 1), [False] * (n + 1))
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = (math.inf, 0)
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = (cur, j0)
                    if minv[j] < delta:
                        delta, j1 = (minv[j], j)
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sum((-cost[p[j] - 1][j - 1] for j in range(1, n + 1)))


def score_calls(pred, reference, mode, numeric_equivalence=False):
    gold = reference["calls"]
    if reference["kind"] == "no_call":
        return float(not pred)
    if not pred or not gold:
        return 0.0
    if mode == "strict":
        return float(
            Counter((call_key(c, numeric_equivalence) for c in pred))
            == Counter((call_key(c, numeric_equivalence) for c in gold))
        )
    if mode != "partial":
        raise ValueError(mode)
    weights = [[pair_score(p, g, numeric_equivalence) for g in gold] for p in pred]
    return 2 * maximum_assignment(weights) / (len(pred) + len(gold))


CALL = re.compile(
    "<tool_call>\\s*<function=([^<>\\n]+)>(.*?)</function>\\s*</tool_call>", re.S
)
PARAM = re.compile("\\s*<parameter=([^<>\\n]+)>(.*?)</parameter>", re.S)
MARKUP = re.compile("</?(?:tool|function|parameter)", re.I)


def strict_xml(text):
    if not text.strip():
        raise ValueError("empty")
    calls, end = ([], 0)
    for m in CALL.finditer(text):
        if MARKUP.search(text[end : m.start()]):
            raise ValueError("unparsed tool markup")
        body, args, pos = (m[2], {}, 0)
        for pm in PARAM.finditer(body):
            if body[pos : pm.start()].strip():
                raise ValueError("text outside parameter")
            if pm[1] in args:
                raise ValueError("duplicate parameter")
            if MARKUP.search(pm[2]):
                raise ValueError("nested tool markup")
            value = pm[2]
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            args[pm[1]] = value
            pos = pm.end()
        if body[pos:].strip():
            raise ValueError("malformed parameter section")
        calls.append((m[1], args))
        end = m.end()
    if MARKUP.search(text[end:]):
        raise ValueError("unparsed tool markup")
    if not calls and (
        re.match("^\\s*\\[\\s*[^\\n]+\\(", text)
        or re.search('"(?:tool_calls|arguments)"\\s*:', text)
    ):
        raise ValueError("foreign call serialization")
    return calls


class NativeParser:

    def __init__(self, tokenizer):
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        self.parser = Qwen3CoderToolParser(tokenizer)

    def parse(self, text, tools):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        raw = strict_xml(text)
        request = ChatCompletionRequest.model_validate(
            {
                "model": "Qwen/Qwen3.5-2B",
                "messages": [{"role": "user", "content": "parse"}],
                "tools": tools,
            }
        )
        self.parser.tools = request.tools
        parsed = self.parser.extract_tool_calls(text, request)
        calls = [
            {"name": c.function.name, "arguments": json.loads(c.function.arguments)}
            for c in parsed.tool_calls
        ]
        if len(calls) != len(raw):
            raise ValueError("native parser recovered only part of markup")
        for c, (name, params) in zip(calls, raw):
            if c["name"] != name or set(c["arguments"]) != set(params):
                raise ValueError("native parser disagreement")
        return calls


def score_text(
    text, tools, reference, parser, mode, truncated=False, numeric_equivalence=False
):
    if truncated:
        return (0.0, {"valid": False, "error": "truncated", "calls": []})
    try:
        strict_xml(text)
        calls = parser.parse(text, tools)
        score = score_calls(calls, reference, mode, numeric_equivalence)
        return (score, {"valid": True, "calls": calls})
    except (ValueError, TypeError, KeyError, OverflowError) as e:
        return (0.0, {"valid": False, "error": str(e), "calls": []})


def make_reward(
    tokenizer,
    mode,
    max_completion_length,
    numeric_equivalence=False,
    model_eos_ids=None,
):
    parser = NativeParser(tokenizer)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if model_eos_ids is not None:
        eos_ids.update(
            model_eos_ids if isinstance(model_eos_ids, list) else [model_eos_ids]
        )

    def reward(
        prompts, completions, completion_ids, reference_json, tools_json, **kwargs
    ):
        scores = []
        for text, ids, ref, tools in zip(
            completions, completion_ids, reference_json, tools_json, strict=True
        ):
            truncated = not ids or ids[-1] not in eos_ids
            content_ids = ids[:-1] if ids and ids[-1] in eos_ids else ids
            text = tokenizer.decode(content_ids, skip_special_tokens=False)
            score, _ = score_text(
                text,
                json.loads(tools),
                json.loads(ref),
                parser,
                mode,
                truncated,
                numeric_equivalence,
            )
            scores.append(score)
        return scores

    reward.__name__ = mode + "_call_reward"
    return reward


SHARED_CONFIG = {
    "seed": 42,
    "max_steps": 32,
    "checkpoint_steps": 16,
    "validation_selection_size": 64,
    "learning_rate": 1e-05,
    "lora_r": 16,
    "lora_alpha": 32,
    "use_vllm": True,
    "numeric_equivalence": True,
}
VARIANTS = {
    "A_strict": {"reward": "strict", "beta": 0.02},
    "C_partial_kl": {"reward": "partial", "beta": 0.1},
    "B_partial": {"reward": "partial", "beta": 0.02},
}
CONFIGS = {
    name: {**SHARED_CONFIG, "name": name, **values} for name, values in VARIANTS.items()
}
VERSIONS = {
    "transformers": "5.9.0",
    "trl": "1.16.0",
    "peft": "0.18.1",
    "vllm": "0.22.0",
    "datasets": "4.7.0",
    "accelerate": "1.10.1",
    "numpy": "2.2.6",
    "bfcl-eval": "2026.3.23",
}


def check_versions():
    actual = {k: version(k) for k in VERSIONS}
    if {k: v.split("+")[0] for k, v in actual.items()} != VERSIONS:
        raise RuntimeError(f"Expected {VERSIONS}, found {actual}")
    return actual


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dataset(rows, tokenizer):
    from datasets import Dataset

    records = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(
            r["prompt"],
            tools=r["tools"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) != r["prompt_tokens"] or len(ids) > 3072:
            raise ValueError("Prompt changed since EDA")
        records.append(
            {
                "prompt": prompt,
                "reference_json": json.dumps(r["reference"], ensure_ascii=False),
                "tools_json": json.dumps(r["tools"], ensure_ascii=False),
                "dialogue": r["dialogue"],
                "turn": r["turn"],
            }
        )
    return Dataset.from_list(records)


def train(variant, output, data_dir=None):
    output = Path(output)
    data_dir = Path(data_dir) if data_dir else ROOT / "data/processed"
    cfg = CONFIGS[variant].copy()
    versions = check_versions()
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from huggingface_hub import snapshot_download
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, revision=MODEL_REVISION, padding_side="left"
    )
    train_rows = load_rows(data_dir / "train.jsonl")
    val_rows = load_rows(data_dir / "validation.jsonl")
    val_rows = sorted(
        val_rows,
        key=lambda r: digest(["validation-selection-42", r["dialogue"], r["turn"]]),
    )[: cfg["validation_selection_size"]]
    train_data = dataset(train_rows, tokenizer)
    val_data = dataset(val_rows, tokenizer)
    base_path = snapshot_download(BASE, revision=MODEL_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        base_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.")
    ]
    if not targets:
        raise RuntimeError("Language-model LoRA targets not found")
    trainable_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    primary = make_reward(
        tokenizer,
        cfg["reward"],
        1024,
        cfg["numeric_equivalence"],
        model.generation_config.eos_token_id,
    )
    common = make_reward(
        tokenizer,
        "strict",
        1024,
        cfg["numeric_equivalence"],
        model.generation_config.eos_token_id,
    )
    common.__name__ = "common_strict"
    params = GRPOConfig(
        output_dir=str(output),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        num_generations=4,
        num_generations_eval=4,
        max_completion_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=cfg["beta"],
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        loss_type="grpo",
        scale_rewards="group",
        reward_weights=[1.0, 0.0],
        mask_truncated_completions=True,
        use_vllm=cfg["use_vllm"],
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=4096,
        vllm_enable_sleep_mode=True,
        eval_strategy="steps",
        eval_steps=cfg["checkpoint_steps"],
        save_strategy="steps",
        save_steps=cfg["checkpoint_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/common_strict/mean",
        greater_is_better=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        warmup_steps=0,
        disable_dropout=True,
        log_completions=False,
    )

    class Log(TrainerCallback):

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                if logs and any(
                    (
                        not math.isfinite(float(logs[k]))
                        for k in ("loss", "grad_norm", "reward")
                        if k in logs
                    )
                ):
                    raise RuntimeError("Non-finite GRPO training metric")
                with (Path(args.output_dir) / "history.jsonl").open("a") as f:
                    f.write(
                        json.dumps({"step": state.global_step, **(logs or {})}) + "\n"
                    )

    meta = {
        "versions": {**versions, "torch": torch.__version__},
        "base": BASE,
        "revision": MODEL_REVISION,
        "train_rows": len(train_rows),
        "validation_selection_ids": [[r["dialogue"], r["turn"]] for r in val_rows],
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [data_dir / "train.jsonl", data_dir / "validation.jsonl"]
        },
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "lora_targets": targets,
        "numeric_comparison": (
            "typed" if not cfg["numeric_equivalence"] else "int_float_equivalent"
        ),
    }
    (output / "environment.json").write_text(json.dumps(meta, indent=2) + "\n")
    trainer = GRPOTrainer(
        model=model,
        args=params,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        peft_config=trainable_cfg,
        reward_funcs=[primary, common],
        callbacks=[Log()],
    )
    start = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    metrics = {
        **result.metrics,
        "elapsed_seconds": time.monotonic() - start,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def merge(adapter, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is not empty")
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base = snapshot_download("Qwen/Qwen3.5-2B", revision=MODEL_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    (output / "merge_provenance.json").write_text(
        json.dumps(
            {
                "base": "Qwen/Qwen3.5-2B",
                "revision": MODEL_REVISION,
                "adapter": str(Path(adapter).resolve()),
            },
            indent=2,
        )
        + "\n"
    )


def validate_model(model, output, data=None):
    output = Path(output)
    data = Path(data) if data else ROOT / "data/processed/validation.jsonl"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-2B", revision=MODEL_REVISION
    )
    rows = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                r["prompt"],
                tools=r["tools"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        }
        for r in rows
    ]
    llm = LLM(
        model=model,
        revision=MODEL_REVISION if model == "Qwen/Qwen3.5-2B" else None,
        tokenizer="Qwen/Qwen3.5-2B",
        tokenizer_revision=MODEL_REVISION,
        generation_config="vllm",
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        seed=42,
    )
    generated = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=1024, skip_special_tokens=False),
    )
    parser = NativeParser(tokenizer)
    results = []
    for r, g in zip(rows, generated, strict=True):
        o = g.outputs[0]
        strict, info = score_text(
            o.text,
            r["tools"],
            r["reference"],
            parser,
            "strict",
            o.finish_reason == "length",
            True,
        )
        partial, _ = score_text(
            o.text,
            r["tools"],
            r["reference"],
            parser,
            "partial",
            o.finish_reason == "length",
            True,
        )
        results.append(
            {
                "dialogue": r["dialogue"],
                "turn": r["turn"],
                "response": o.text,
                "finish_reason": o.finish_reason,
                "strict": strict,
                "partial": partial,
                **info,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text(
        "".join((json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "n": len(results),
                "strict": mean((r["strict"] for r in results)),
                "partial": mean((r["partial"] for r in results)),
                "invalid": sum((not r["valid"] for r in results)),
            },
            indent=2,
        )
        + "\n"
    )


MATERIAL_COMMIT = "8888f2d8492974f0173f44e6faa05742b9a71344"
MATERIAL_HASHES = {
    "bfcl.jsonl": "b257ffe2c539897123706ed2211f9ad0e68b8f8fbad7b523ba5a3fcca2a457b2",
    "evaluate.py": "62411a6f1bd790bd54f27fb09e567075b5b27510d126ebe145e72b8319abc7b6",
}


def materials():
    folder = ROOT / "data/benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in MATERIAL_HASHES.items():
        p = folder / name
        if not p.exists():
            url = f"https://raw.githubusercontent.com/anbelk/llm-fcs-2026/{MATERIAL_COMMIT}/hw5_rl/materials/{name}"
            urllib.request.urlretrieve(url, p)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Course file hash mismatch: {name}")
    return folder


def prepare(data=None, output=None, tokenizer=BASE):
    data = Path(data) if data else ROOT / "data/toolace.json"
    output = Path(output) if output else ROOT / "data/processed"
    benchmark = materials() / "bfcl.jsonl"
    prepare_data(data, output, tokenizer, benchmark)


def check():
    pred = [{"name": "f", "arguments": {"x": 1}}] * 2
    gold = {"kind": "calls", "calls": [{"name": "f", "arguments": {"x": 1.0}}]}
    assert score_calls(pred, gold, "strict", True) == 0
    assert abs(score_calls(pred, gold, "partial", True) - 2 / 3) < 1e-12
    assert typed_key(True, True) != typed_key(1, True)
    assert maximum_assignment([[0.9, 0.8], [0.85, 0.1]]) == 1.65
    groups = {
        "simple": ["simple"],
        "multiple": ["multiple", "live_multiple"],
        "parallel": ["parallel"],
        "parallel_multiple": ["parallel_multiple"],
        "irrelevance": ["irrelevance", "live_irrelevance"],
    }
    for name in ("baseline", "A_strict", "B_partial", "C_partial_kl"):
        folder = ROOT / "outputs/bfcl" / name
        rows = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text().splitlines()
        ]
        metric = json.loads((folder / "metrics.json").read_text())
        assert len(rows) == 350
        categories = defaultdict(list)
        for row in rows:
            categories[row["category"]].append(row["check"]["valid"])
        score = mean(
            mean(mean(categories[c]) for c in group) for group in groups.values()
        )
        assert abs(score - metric["overall"]) < 1e-12
    print("Награды и метрики четырёх моделей совпали.")


def run(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = materials()
    check_versions()
    from huggingface_hub import snapshot_download

    base = snapshot_download(BASE, revision=MODEL_REVISION)
    script = Path(__file__).resolve()

    def execute(name, command, cwd=ROOT):
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(
                [str(v) for v in command],
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def evaluate(name, model):
        execute(
            name + "_bfcl",
            [
                sys.executable,
                benchmark / "evaluate.py",
                model,
                "--output",
                output / name / "bfcl",
            ],
            benchmark,
        )
        execute(
            name + "_validation",
            [
                sys.executable,
                script,
                "validation",
                "--model",
                model,
                "--output",
                output / name / "validation",
            ],
        )

    evaluate("baseline", base)
    for name in ("A_strict", "B_partial", "C_partial_kl"):
        folder = output / name
        execute(
            name + "_train",
            [
                sys.executable,
                script,
                "train",
                "--variant",
                name,
                "--output",
                folder / "training",
            ],
        )
        execute(
            name + "_merge",
            [
                sys.executable,
                script,
                "merge",
                "--adapter",
                folder / "training/adapter",
                "--output",
                folder / "merged",
            ],
        )
        evaluate(name, folder / "merged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--tokenizer", default=BASE)
    p = commands.add_parser("train")
    p.add_argument("--variant", choices=CONFIGS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir")
    p = commands.add_parser("merge")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("validation")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data")
    p = commands.add_parser("run")
    p.add_argument("--output", required=True)
    commands.add_parser("check")
    args = vars(parser.parse_args())
    command = args.pop("command")
    functions = {
        "prepare": prepare,
        "train": train,
        "merge": merge,
        "validation": validate_model,
        "run": run,
        "check": check,
    }
    functions[command](**args)


if __name__ == "__main__":
    main()

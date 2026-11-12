"""ToolACE audit and conservative RL decision extraction; no model weights or training.

Run with --help. Source values are never evaluated as executable Python.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import urllib.request

REVISION = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
URL = f'https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/{REVISION}/data.json'
SOURCE_SHA256 = 'ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e'


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def serialized(x):
    # Keep schema/property and historical argument order exactly as tokenized.
    # Canonical sorting is only for hashes; Qwen's template can observe dict order.
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def norm(s):
    return ' '.join(s.casefold().split())


def parts(s, separator=','):
    """Split at top level, respecting quotes, escapes and nested literals."""
    out, stack, quote, escaped, start = [], [], None, False, 0
    pairs = {')': '(', ']': '[', '}': '{'}
    for i, c in enumerate(s):
        if quote:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == quote:
                quote = None
        elif c in '\"\'':
            quote = c
        elif c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack.pop() != pairs[c]:
                raise ValueError('unbalanced delimiters')
        elif c == separator and not stack:
            out.append(s[start:i].strip())
            start = i + 1
    if quote or stack:
        raise ValueError('unclosed literal')
    out.append(s[start:].strip())
    return out


def literal(s):
    try:
        x = json.loads(s)
        json.dumps(x, allow_nan=False)
        return x
    except (ValueError, TypeError):
        x = ast.literal_eval(s)
        # Reject sets, bytes, complex, tuple and non-string dict keys: not JSON arguments.
        def check(v):
            if type(v) in (str, int, float, bool, type(None)):
                return
            if isinstance(v, list):
                for e in v: check(e)
            elif isinstance(v, dict) and all(isinstance(k, str) for k in v):
                for e in v.values(): check(e)
            else:
                raise ValueError('non-JSON literal')
        check(x)
        json.dumps(x, allow_nan=False)
        return x


def parse_calls(text, names):
    text = text.strip()
    if not text.startswith('['):
        return None  # natural-language answer, NOT a parse failure
    if not text.endswith(']'):
        raise ValueError('bracket prefix without closing bracket')
    if not text[1:-1].strip():
        return []
    calls, body_text, pos = [], text[1:-1], 0
    while pos < len(body_text):
        chunk = body_text[pos:].lstrip()
        candidates = [n for n in names if chunk.startswith(n + '(')]
        if not candidates:
            raise ValueError('unknown function or non-call bracket text')
        name = max(candidates, key=len)
        # Function names are opaque schema strings; only parse delimiters AFTER the name.
        start, stack, quote, escaped, end = len(name), 0, None, False, None
        for i in range(start, len(chunk)):
            c = chunk[i]
            if quote:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == quote: quote = None
            elif c in '\"\'': quote = c
            elif c == '(': stack += 1
            elif c == ')':
                stack -= 1
                if stack == 0:
                    end = i; break
        if end is None:
            raise ValueError('call missing closing parenthesis')
        body = chunk[len(name)+1:end].strip()
        args = {}
        for arg in parts(body) if body else []:
            kv = parts(arg, '=')
            if len(kv) != 2 or not kv[0] or kv[0] in args:
                raise ValueError('bad or duplicate argument')
            args[kv[0]] = literal(kv[1])
        calls.append({'name': name, 'arguments': args})
        rest = chunk[end+1:].strip()
        if not rest: break
        if not rest.startswith(',') or not rest[1:].strip():
            raise ValueError('missing call separator or trailing comma')
        body_text, pos = rest[1:].lstrip(), 0
    return calls


def convert_schema(x):
    if isinstance(x, list):
        return [convert_schema(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {k: convert_schema(v) for k, v in x.items() if not (k == 'required' and v is None)}
    if isinstance(out.get('type'), str):
        out['type'] = {'dict': 'object', 'int': 'integer', 'float': 'number', 'tuple': 'array', 'any': 'string'}.get(out['type'], out['type'])
    return out


def type_ok(x, t):
    checks = {'string': lambda: isinstance(x, str), 'integer': lambda: type(x) is int, 'int': lambda: type(x) is int,
              'number': lambda: type(x) in (int, float), 'float': lambda: type(x) in (int, float),
              'boolean': lambda: type(x) is bool, 'dict': lambda: isinstance(x, dict),
              'object': lambda: isinstance(x, dict), 'array': lambda: isinstance(x, list),
              'tuple': lambda: isinstance(x, list), 'null': lambda: x is None,
              'any': lambda: True}
    if isinstance(t, list):
        return any(type_ok(x, v) for v in t)
    return checks.get(t, lambda: True)()


def validate(x, schema, path=''):
    issues = []
    if not type_ok(x, schema.get('type')):
        return [('type_mismatch', path)]
    if 'enum' in schema and x not in schema['enum']:
        issues.append(('enum_mismatch', path))
    if isinstance(x, dict):
        props = schema.get('properties', {})
        for k in schema.get('required') or []:
            if k not in x: issues.append(('missing_required', path + '.' + k))
        for k, v in x.items():
            if k not in props and ('properties' in schema or schema.get('additionalProperties') is False):
                # Undeclared argument flagged conservatively, even if additionalProperties unspecified.
                issues.append(('undeclared_argument', path + '.' + k))
            elif k in props:
                issues.extend(validate(v, props[k], path + '.' + k))
    if isinstance(x, list) and isinstance(schema.get('items'), dict):
        for i, v in enumerate(x): issues.extend(validate(v, schema['items'], path + f'[{i}]'))
    return issues


def dist(values):
    values = sorted(values)
    if not values: return {}
    return {'n': len(values), 'min': values[0], 'p50': values[int((len(values)-1)*.5)],
            'p90': values[int((len(values)-1)*.9)], 'p95': values[int((len(values)-1)*.95)],
            'p99': values[int((len(values)-1)*.99)], 'max': values[-1],
            'mean': round(sum(values)/len(values), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokenizer', default='Qwen/Qwen3.5-2B')
    ap.add_argument('--bfcl', type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, args.data)
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('ToolACE hash differs from the audited revision')
    data = json.loads(raw)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION)
    counts, roles, tools_hist, user_hist, call_hist = (Counter() for _ in range(5))
    types, value_types, issues_count, seqs, argument_hist, unsupported_formats = (Counter() for _ in range(6))
    source_chars, source_tokens, prompt_tokens, target_tokens = [], [], [], []
    candidates, exceptions, all_users, full_hashes, tool_sets = [], [], [], [], []
    schema_hashes, function_names = set(), set()
    for di, row in enumerate(data):
        full_hashes.append(digest(row))
        conv = row['conversations']
        source = row['system'] + '\n' + '\n'.join(m['value'] for m in conv)
        source_chars.append(len(source)); source_tokens.append(len(tokenizer.encode(source)))
        roles.update(m['from'] for m in conv)
        counts['all_assistant_turns'] += sum(m['from'] == 'assistant' for m in conv)
        counts['all_decisions_after_user'] += sum(m['from'] == 'assistant' and j>0 and conv[j-1]['from']=='user' for j,m in enumerate(conv))
        seqs['/'.join(m['from'] for m in conv)] += 1
        user_hist[sum(m['from'] == 'user' for m in conv)] += 1
        users = [norm(m['value']) for m in conv if m['from'] == 'user']
        all_users.append(users)
        try:
            markers = ['Here is a list of functions in JSON format that you can invoke:', '以下是可以调用的工具的JSON格式列表:']
            marker = next((s for s in markers if s in row['system']), None)
            if marker is None:
                raise ValueError('unsupported schema/call representation; not asserted to be corrupt')
            pos = row['system'].index('[', row['system'].index(marker))
            tools, _ = json.JSONDecoder().raw_decode(row['system'][pos:])
            assert isinstance(tools, list) and all('name' in t and 'parameters' in t for t in tools)
        except Exception as e:
            counts['schema_parse_fail_dialogues'] += 1
            for label, pattern in {'html':'<table', 'xml':'<tool_name>', 'latex':'\\begin{tabular}',
                                   'markdown':'**tool_name**', 'json_object':'{"tool_name"',
                                   'yaml_or_plain':'tool_name:'}.items():
                unsupported_formats[label] += pattern in row['system']
            exceptions.append({'dialogue': di, 'reason': 'schema_parse', 'detail': str(e)})
            tool_sets.append(set()); continue
        by_name = {t['name']: t for t in tools}
        tools_hist[len(tools)] += 1
        tool_sets.append(set(by_name))
        function_names.update(by_name)
        for t in tools:
            schema_hashes.add(digest(t))
            types.update(str(p.get('type')) for p in t['parameters'].get('properties', {}).values())
            if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', t['name']): counts['non_python_name_occurrences'] += 1
        if len(by_name) != len(tools): counts['duplicate_names_in_schema_dialogues'] += 1
        api_tools = [{'type': 'function', 'function': convert_schema(t)} for t in tools]
        prefix = row['system'].split(marker)[0]
        time_context = '\n'.join(line.strip() for line in prefix.splitlines()
                                 if re.search(r'\b(today|current time|current date)\b', line, re.I))
        chinese_time = re.search(r'今天是[^。\n]*。',prefix)
        if chinese_time: time_context += chinese_time.group()
        counts['dialogues_with_time_context'] += bool(time_context)
        history = [{'role':'system', 'content':time_context}] if time_context else []
        history_bad = False
        for ti, m in enumerate(conv):
            role, text = m['from'], m['value']
            if role != 'assistant':
                history.append({'role': role, 'content': text}); continue
            counts['assistant_turns'] += 1
            after_user = ti > 0 and conv[ti-1]['from'] == 'user'
            counts['after_user' if after_user else 'not_after_user'] += 1
            err, call_issues = None, []
            try:
                calls = parse_calls(text, by_name)
                if calls is None and ti+1 < len(conv) and conv[ti+1]['from'] == 'tool':
                    raise ValueError('tool response follows non-call answer')
                for c in calls or []:
                    call_issues.extend(validate(c['arguments'], by_name[c['name']]['parameters']))
                    counts['calls'] += 1
                    counts['arguments'] += len(c['arguments'])
                    argument_hist[len(c['arguments'])] += 1
                    value_types.update(type(v).__name__ for v in c['arguments'].values())
                    for k, p in by_name[c['name']]['parameters'].get('properties', {}).items():
                        if k not in c['arguments'] and k not in (by_name[c['name']]['parameters'].get('required') or []):
                            counts['omitted_optional_arguments'] += 1
            except (ValueError, SyntaxError, TypeError, KeyError, AssertionError) as e:
                calls, err = None, str(e)
                counts['call_parse_fail_turns'] += 1
            if err or call_issues:
                exceptions.append({'dialogue': di, 'turn': ti, 'reason': 'call_parse' if err else 'schema_validation',
                                   'detail': err or call_issues, 'source': text})
            issues_count.update(k for k, _ in call_issues)
            counts['turns_with_schema_issues'] += bool(call_issues)
            if not err:
                call_hist[len(calls or [])] += 1
                if calls and len({c['name'] for c in calls}) < len(calls):
                    counts['turns_with_repeated_function_name'] += 1
                counts['calls_turns' if calls else 'text_or_empty_turns'] += 1
                if after_user: counts['decision_call_turns' if calls else 'decision_no_call_turns'] += 1
                else: counts['post_tool_call_turns' if calls else 'post_tool_no_call_turns'] += 1
            if after_user:
                reason = ('bad_current_target' if err or call_issues else
                          'bad_history' if history_bad else
                          'duplicate_schema_names' if len(by_name) != len(tools) else None)
                if reason:
                    counts['excluded_' + reason] += 1
                else:
                    messages = list(history)
                    # No original ToolACE system: it demands incompatible Python-like serialization.
                    tokens = tokenizer.apply_chat_template(messages, tools=api_tools, tokenize=True,
                               return_dict=False, add_generation_prompt=True, enable_thinking=False)
                    p_len = len(tokens)
                    target_message = {'role': 'assistant', 'content': text if not calls else '',
                                      **({'tool_calls': [{'type': 'function', 'function': c} for c in calls]} if calls else {})}
                    completed = tokenizer.apply_chat_template(messages+[target_message], tools=api_tools,
                                   tokenize=True, return_dict=False, enable_thinking=False)
                    # Count the actual template suffix rather than a standalone raw target.
                    assert completed[:p_len] == tokens, 'prompt is not prefix of completed template'
                    t_len = len(completed)-p_len
                    prompt_tokens.append(p_len); target_tokens.append(t_len)
                    ref = {'kind': 'calls' if calls else 'no_call', 'calls': calls or []}
                    candidates.append({'dialogue': di, 'turn': ti, 'prompt': messages, 'tools': api_tools,
                                       'reference': ref, 'prompt_tokens': p_len, 'target_tokens': t_len,
                                       'prompt_hash': digest([api_tools, messages]), 'reference_hash': digest(ref),
                                       'fits_4096_with_1024_reserve': p_len <= 3072,
                                       'reference_fits_1024': t_len <= 1024})
            history_bad |= bool(err or call_issues)
            if calls and not err:
                history.append({'role': 'assistant', 'content': '',
                                'tool_calls': [{'type': 'function', 'function': c} for c in calls]})
            else:
                history.append({'role': 'assistant', 'content': text})
    # Union exact dialogue duplicates and exact prompts, without ever splitting one dialogue.
    parent = list(range(len(data)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a,b)] = min(a,b)
    seen = {}
    for i, h in enumerate(full_hashes):
        if h in seen: union(i, seen[h])
        else: seen[h] = i
    prompt_sources, prompt_refs = {}, defaultdict(set)
    for c in candidates:
        h = c['prompt_hash']; prompt_refs[h].add(c['reference_hash'])
        if h in prompt_sources: union(c['dialogue'], prompt_sources[h])
        else: prompt_sources[h] = c['dialogue']
    groups = defaultdict(list)
    for i in range(len(data)): groups[root(i)].append(i)
    split = {}
    for ids in groups.values():
        group_key = min(full_hashes[i] for i in ids)
        name = 'validation' if int(digest(['42', group_key]),16) % 10 == 0 else 'train'
        for i in ids: split[i] = name
    selected, dedup_seen = [], set()
    for c in candidates:
        if len(prompt_refs[c['prompt_hash']]) > 1:
            counts['excluded_conflicting_prompt_targets'] += 1; continue
        if c['prompt_hash'] in dedup_seen:
            counts['excluded_duplicate_prompts'] += 1; continue
        dedup_seen.add(c['prompt_hash'])
        c['split'] = split[c['dialogue']]
        if not c['fits_4096_with_1024_reserve'] or not c['reference_fits_1024']:
            counts['excluded_length'] += 1; continue
        selected.append(c)
    user_sets = {s: {u for i, us in enumerate(all_users) if split[i] == s for u in us} for s in ('train','validation')}
    names_sets = {s: set().union(*(ts for i, ts in enumerate(tool_sets) if split[i] == s)) for s in ('train','validation')}
    leak = {'shared_normalized_user_utterances': len(user_sets['train'] & user_sets['validation']),
            'shared_function_names': len(names_sets['train'] & names_sets['validation'])}
    if args.bfcl:
        bfcl = [json.loads(s) for s in args.bfcl.read_text().splitlines() if s.strip()]
        bu = {norm(m['content']) for b in bfcl for m in b['question'] if m['role'] == 'user'}
        bn = {t['name'] for b in bfcl for t in b['function']}
        leak.update(bfcl_rows=len(bfcl), bfcl_sha256=hashlib.sha256(args.bfcl.read_bytes()).hexdigest(),
                    bfcl_exact_normalized_user_overlap=len(bu & set().union(*user_sets.values())),
                    bfcl_function_name_overlap=len(bn & function_names))
    for s in ('train','validation'):
        rows = [c for c in selected if c['split'] == s]
        (args.out / f'{s}.jsonl').write_text(''.join(serialized(c)+'\n' for c in rows))
    (args.out/'candidates.jsonl').write_text(''.join(serialized(c)+'\n' for c in candidates))
    (args.out/'exceptions.jsonl').write_text(''.join(serialized(c)+'\n' for c in exceptions))
    (args.out/'split_manifest.json').write_text(canonical({'seed':42,'dialogue_split':split,'groups':list(groups.values())})+'\n')
    counts.update(dialogues=len(data), unique_full_dialogues=len(set(full_hashes)),
                  unique_function_names=len(function_names), unique_function_schemas=len(schema_hashes),
                  candidate_decisions=len(candidates), selected_decisions=len(selected))
    report = {'source': {'url':URL, 'revision':REVISION, 'sha256':hashlib.sha256(raw).hexdigest()},
              'environment': {'python': platform.python_version(), **{p:version(p) for p in ('transformers','tokenizers','jinja2')}},
              'model_revision':MODEL_REVISION, 'counts':counts, 'roles':roles, 'role_sequences':seqs,
              'available_tools_per_dialogue':tools_hist,'user_turns_per_dialogue':user_hist,
              'calls_per_assistant_turn':call_hist, 'declared_parameter_types_occurrences':types,
              'arguments_per_call':argument_hist, 'unsupported_schema_format_markers_nonexclusive':unsupported_formats,
              'observed_argument_value_types':value_types,
              'schema_issues':{k:issues_count[k] for k in ('undeclared_argument','missing_required','type_mismatch','enum_mismatch')},
              'lengths': {'source_dialogue_chars':dist(source_chars),'source_dialogue_tokens_no_template':dist(source_tokens),
                          'candidate_prompt_tokens':dist(prompt_tokens),'candidate_target_tokens':dist(target_tokens),
                          'candidate_prompt_gt3072':sum(n>3072 for n in prompt_tokens),
                          'candidate_prompt_gt4096':sum(n>4096 for n in prompt_tokens),
                          'candidate_reference_gt1024':sum(n>1024 for n in target_tokens)},
              'duplicates': {'full_dialogue_extra_copies':len(data)-len(set(full_hashes)),
                             'candidate_prompt_extra_copies':len(candidates)-len(prompt_refs),
                             'conflicting_prompt_groups':sum(len(v)>1 for v in prompt_refs.values())},
              'split': {s:{'dialogues':sum(v==s for v in split.values()),
                           'dialogues_contributing_selected':len({c['dialogue'] for c in selected if c['split']==s}),
                           'selected_decisions':sum(c['split']==s for c in selected),
                           'selected_no_call':sum(c['split']==s and c['reference']['kind']=='no_call' for c in selected)}
                        for s in ('train','validation')}, 'overlap':leak}
    (args.out/'stats.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

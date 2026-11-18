import ast
import collections
import hashlib
import io
import json
import keyword
import re
import tokenize
from pathlib import Path

import typer

SCHEMA_MARKER = 'Here is a list of functions in JSON format that you can invoke:'
SCHEMA_DICT_KEYS = {'properties', 'patternProperties', '$defs', 'definitions'}


def normalize_schema(schema):
    if isinstance(schema, list):
        return [normalize_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    normalized = {}
    for k, v in schema.items():
        if k == 'type' and isinstance(v, str):
            normalized[k] = {'dict': 'object', 'int': 'integer', 'float': 'number',
                             'bool': 'boolean', 'list': 'array', 'tuple': 'array',
                             'str': 'string', 'any': 'string'}.get(v, v)
        elif k in SCHEMA_DICT_KEYS and isinstance(v, dict):
            normalized[k] = {name: normalize_schema(spec) for name, spec in v.items()}
        elif k not in {'default', 'enum', 'examples', 'const'}:
            normalized[k] = normalize_schema(v)
        else:
            normalized[k] = v
    return normalized


def read_tools(system):
    if SCHEMA_MARKER not in system:
        raise ValueError('missing tool schema')
    raw, _ = json.JSONDecoder().raw_decode(system.split(SCHEMA_MARKER, 1)[1].lstrip())
    if not isinstance(raw, list) or not raw:
        raise ValueError('empty tool schema')
    tools, aliases = [], {}
    used = set()
    for spec in raw:
        if not isinstance(spec, dict) or not isinstance(spec.get('name'), str):
            raise ValueError('invalid tool name')
        name = spec['name']
        if name in aliases or not isinstance(spec.get('parameters'), dict):
            raise ValueError('ambiguous tool schema')
        base = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')[:48] or 'tool'
        alias = base
        if alias in used:
            alias = f'{base}_{hashlib.sha256(name.encode()).hexdigest()[:10]}'
        if alias in used:
            raise ValueError('alias collision')
        aliases[name] = alias
        used.add(alias)
        tools.append({'type': 'function', 'function': {
            'name': alias, 'description': spec.get('description', ''),
            'parameters': normalize_schema(spec['parameters']),
        }})
    return tools, aliases


def parse_calls(text, aliases):
    remaining = text.strip()[1:-1].strip()
    calls = []
    while remaining:
        name, opening, arg_source = remaining.partition('(')
        name = name.strip()
        if not opening or name not in aliases:
            raise ValueError('unknown function')
        source = opening + arg_source
        depth = 0
        tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            tok_type, value = tok.type, tok.string
            if tok_type == tokenize.OP and value in '([{':
                depth += 1
            elif tok_type == tokenize.OP and value in ')]}':
                depth -= 1
            tokens.append((tok_type, value))
            if depth == 0:
                end_line, end_column = tok.end
                break
        if depth or tokens[-1][1] != ')':
            raise ValueError('unclosed call')
        replaced = {}
        for index, (kind, value) in enumerate(tokens):
            if kind != tokenize.NAME:
                continue
            following = tokens[index + 1][1] if index + 1 < len(tokens) else ''
            if following == '=' and keyword.iskeyword(value):
                placeholder = f'__toolace_keyword_{index}'
                replaced[placeholder] = value
                tokens[index] = (kind, placeholder)
            elif value in {'true', 'false', 'null'}:
                tokens[index] = (kind, {'true': 'True', 'false': 'False', 'null': 'None'}[value])
        expression = ast.parse('f' + tokenize.untokenize(tokens), mode='eval').body
        if not isinstance(expression, ast.Call) or expression.args:
            raise ValueError('expected keyword arguments')
        arguments = {}
        for kw in expression.keywords:
            if kw.arg is None:
                raise ValueError('unpacked argument')
            key = replaced.get(kw.arg, kw.arg)
            if key in arguments:
                raise ValueError('duplicate argument')
            arguments[key] = ast.literal_eval(kw.value)
        arguments = json.loads(json.dumps(arguments, allow_nan=False))
        calls.append({'name': aliases[name], 'arguments': arguments})
        lines = source.splitlines(keepends=True)
        remaining = (lines[end_line - 1][end_column:] + ''.join(lines[end_line:])).strip()
        if remaining:
            if not remaining.startswith(',') or not remaining[1:].strip():
                raise ValueError('invalid call separator')
            remaining = remaining[1:].strip()
    if not calls:
        raise ValueError('empty call list')
    return calls


def category(tools, calls):
    if not calls:
        return 'no_call'
    if len(calls) == 1:
        return 'simple' if len(tools) == 1 else 'multiple'
    return 'parallel' if len({call['name'] for call in calls}) == 1 else 'parallel_multiple'


def filter_dialogue(row, counts):
    try:
        return read_tools(row['system'])
    except (ValueError, KeyError, TypeError):
        counts['skipped_dialogue_schema'] += 1
        return None


def reference_calls(text, aliases, next_is_tool):
    if not text.startswith('[') and not next_is_tool:
        return []
    if not (text.startswith('[') and text.endswith(']')):
        raise ValueError('unknown call syntax')
    return parse_calls(text, aliases)


def prepare(rows, output, seed):
    counts = collections.Counter()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'train.jsonl').open('w') as train, (output / 'validation.jsonl').open('w') as validation:
        for index, row in enumerate(rows):
            schema = filter_dialogue(row, counts)
            if schema is None:
                continue
            tools, aliases = schema
            dialogue_hash = hashlib.sha256(json.dumps(
                row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            bucket = int(hashlib.sha256(f'{seed}:{dialogue_hash}'.encode()).hexdigest()[:8], 16)
            split, writer = ('validation', validation) if bucket % 10 == 0 else ('train', train)
            history = []
            talk = row['conversations']
            for turn, message in enumerate(talk):
                role, value = message['from'], message['value']
                text = value.strip()
                if role == 'assistant':
                    next_is_tool = turn + 1 < len(talk) and talk[turn + 1]['from'] == 'tool'
                    try:
                        calls = reference_calls(text, aliases, next_is_tool)
                    except (ValueError, SyntaxError, TypeError, tokenize.TokenError, NameError, OverflowError):
                        counts['skipped_dialogue_tail_call_syntax'] += 1
                        break
                    if history and history[-1]['role'] == 'user':
                        if text:
                            kind = category(tools, calls)
                            record = {'dialogue_id': dialogue_hash, 'source_row': index,
                                      'turn': turn, 'prompt': history, 'tools': tools,
                                      'expected_calls': calls, 'category': kind}
                            writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                            counts[f'{split}/{kind}'] += 1
                        else:
                            counts['skipped_empty_target'] += 1
                    if calls:
                        history.append({'role': role, 'content': '', 'tool_calls': [
                            {'type': 'function', 'function': call} for call in calls
                        ]})
                        if not next_is_tool and turn + 1 < len(talk):
                            counts['skipped_dialogue_tail_missing_tool_result'] += 1
                            break
                    else:
                        history.append({'role': role, 'content': value})
                elif role == 'tool':
                    history.append({'role': role, 'content': value})
                elif role == 'user':
                    history.append({'role': role, 'content': value})
    (output / 'counts.json').write_text(json.dumps(dict(sorted(counts.items())), indent=2) + '\n')
    print(json.dumps(dict(sorted(counts.items())), indent=2))


def main(source: Path, output: Path, seed: int = 42):
    prepare(json.loads(source.read_text()), output, seed)


if __name__ == '__main__':
    typer.run(main)

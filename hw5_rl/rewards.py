import json
import re
from functools import lru_cache

MAX_TARGET_CALLS = 12


def normalize(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def same_value(left, right, normalize_numbers=True):
    if normalize_numbers:
        left, right = normalize(left), normalize(right)
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False)


def score_calls(expected, predicted, text, valid=True, normalize_numbers=True):
    if not valid or not text.strip():
        return 0.0, 0.0
    if not expected:
        match = re.match(r'^\s*\[\s*[^\[\]"\']{1,128}\(', text)
        valid_no_call = not predicted and not match and not any(
            marker in text for marker in ('<tool', '<function', '<parameter'))
        return (1.0, 1.0) if valid_no_call else (0.0, 0.0)
    if not predicted:
        return 0.0, 0.0

    def key(call):
        arguments = normalize(call['arguments']) if normalize_numbers else call['arguments']
        return call['name'], json.dumps(arguments, sort_keys=True, ensure_ascii=False)

    if sorted(map(key, expected)) == sorted(map(key, predicted)):
        return 1.0, 1.0
    if len(expected) > MAX_TARGET_CALLS or len(predicted) > MAX_TARGET_CALLS:
        return 0.0, 0.0

    def agreement(target, output):
        if target['name'] != output['name']:
            return 0.0
        a, b = target['arguments'], output['arguments']
        shared = sum(same_value(a[k], b[k], normalize_numbers) for k in a.keys() & b.keys())
        denominator = len(a.keys() | b.keys())
        return 0.5 + 0.5 * (shared / denominator if denominator else 1.0)

    weights = [[agreement(target, output) for target in expected]
               for output in predicted]

    @lru_cache(maxsize=None)
    def best(i, used):
        if i == len(weights):
            return 0.0
        result = best(i + 1, used)
        for j, weight in enumerate(weights[i]):
            if weight and not (used & (1 << j)):
                result = max(result, weight + best(i + 1, used | (1 << j)))
        return result

    return 0.0, best(0, 0) / max(len(expected), len(predicted))

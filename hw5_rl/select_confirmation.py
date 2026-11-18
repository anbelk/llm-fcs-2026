import json
import random
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

root = Path('.')
work = Path('outputs/long_rl')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc')
train = [json.loads(x) for x in (root / 'data/train.jsonl').read_text().splitlines()]
valid = [json.loads(x) for x in (root / 'data/validation.jsonl').read_text().splitlines()]
dev = [json.loads(x) for x in (work / 'validation_selection.jsonl').read_text().splitlines()]
ids = {(r['dialogue_id'], r['turn']) for r in dev}
dev_records = [r for r in valid if (r['dialogue_id'], r['turn']) in ids]
assert len(dev_records) == len(dev) == 150

def key(r):
    return json.dumps([r['prompt'], r['tools']], sort_keys=True, ensure_ascii=False)

excluded = {key(r) for r in train + dev_records}
excluded_dialogues = {r['dialogue_id'] for r in dev_records}
candidates = [r for r in valid if r['dialogue_id'] not in excluded_dialogues and key(r) not in excluded]
random.Random(57).shuffle(candidates)
counts = Counter()
selected = []
used_dialogues = set()
used_prompts = set()
for r in candidates:
    prompt = tokenizer.apply_chat_template(r['prompt'], tools=r['tools'], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > 2048:
        continue
    if counts[r['category']] < 30 and r['dialogue_id'] not in used_dialogues and key(r) not in used_prompts:
        selected.append(r)
        counts[r['category']] += 1
        used_dialogues.add(r['dialogue_id'])
        used_prompts.add(key(r))
assert len(selected) == 150 and set(counts.values()) == {30}
assert all(key(r) not in excluded and r['dialogue_id'] not in excluded_dialogues for r in selected)
(root / 'data/confirmation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
print('selected', len(selected), dict(counts), 'unique dialogues', len(used_dialogues), 'train/dev prompt+tools overlap', 0, 'dev dialogue overlap', 0)

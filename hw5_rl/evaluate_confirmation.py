import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from train import MODEL, REVISION, ValidationCallback, load_split


def main(adapter: Path, output: Path):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = 'left'
    data = load_split(Path('data/confirmation.jsonl'), tokenizer, 2048, 150, 57)
    if len(data) != 150:
        raise ValueError('Expected all 150 confirmation examples')
    with (output / 'selection.jsonl').open('w') as f:
        for row in data:
            f.write(json.dumps({k: row[k] for k in ('dialogue_id', 'turn', 'category')}) + '\n')
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to('cuda')
    callback = ValidationCallback(tokenizer, data, output, 1, 10, 512)
    callback.best_score = 2.0
    callback.check(model, 0, SimpleNamespace(should_training_stop=False))
    model = PeftModel.from_pretrained(model, str(adapter), adapter_name='candidate')
    model.set_adapter('candidate')
    callback.check(model, 1, SimpleNamespace(should_training_stop=False))
    rows = [json.loads(x) for x in (output / 'validation_predictions.jsonl').read_text().splitlines()]
    def scores(step):
        return {(r['dialogue_id'], r['turn']): r['score'] for r in rows if r['step'] == step}
    comparisons = {}
    candidate = scores(1)
    for name, step in [('base', 0)]:
        reference = scores(step)
        if not reference:
            continue
        if set(candidate) != set(reference):
            raise ValueError('Validation examples differ')
        gains = sum(candidate[k] > reference[k] for k in reference)
        losses = sum(candidate[k] < reference[k] for k in reference)
        discordant = gains + losses
        p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1)) / 2**discordant) if discordant else 1.0
        comparisons[name] = {'n': len(reference), 'gains': gains, 'losses': losses,
                             'net': gains - losses, 'exact_mcnemar_p': p}
    (output / 'paired.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    typer.run(main)

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer


def verdicts(path):
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    scores = {r['id']: (r['category'], int(r['check']['valid'])) for r in rows}
    if len(scores) != len(rows):
        raise ValueError('Repeated BFCL example')
    return scores


def main(base: Path, candidate: Path, seed: int = 42, repeats: int = 20000):
    before, after = verdicts(base), verdicts(candidate)
    if before.keys() != after.keys():
        raise ValueError('BFCL selections differ')
    strata = defaultdict(list)
    for example, (category, score) in before.items():
        new_category, new_score = after[example]
        if category != new_category:
            raise ValueError(f'Category changed for {example}')
        strata[category].append(new_score - score)
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for category, changes in sorted(strata.items()):
        draws = rng.choice(changes, (repeats, len(changes)), replace=True)
        groups[category.removeprefix('live_')].append(draws.mean(axis=1))
    overall = np.mean([np.mean(parts, axis=0) for parts in groups.values()], axis=0)
    print(np.percentile(100 * overall, [2.5, 97.5]))


if __name__ == '__main__':
    typer.run(main)

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root=Path(__file__).resolve().parents[1]
raw=(root/'bfcl.jsonl').read_bytes()
ids={json.loads(l)['id'] for l in raw.splitlines()}
digest=hashlib.sha256(raw).hexdigest()
assert len(ids)==350
base={}
for directory in ('round1/base', 'round1/rl1-bfcl', 'round1/rl2-bfcl', 'round1/rl3-bfcl', 'long_rl_bfcl'):
    path = root / 'outputs' / directory / 'metrics.json'
    metrics=json.loads(path.read_text())
    if 'benchmark_sha256' not in metrics:
        continue
    assert digest==metrics['benchmark_sha256']
    rows=[json.loads(l) for l in path.with_name('predictions.jsonl').open()]
    assert len(rows)==350 and {r['id'] for r in rows}==ids
    assert metrics['decoding']=={'temperature':0,'max_new_tokens':1024,'enable_thinking':False,'seed':42}
    scores=defaultdict(list)
    verdict={}
    for r in rows:
        scores[r['category'].removeprefix('live_')].append(int(r['check']['valid']))
        verdict[r['id']]=r['check']['valid']
    macro=sum(sum(s)/len(s) for s in scores.values())/len(scores)
    assert len(scores)==5 and abs(macro-metrics['overall'])<1e-9
    assert all(abs(sum(s)/len(s)-metrics['groups'][k])<1e-9 for k,s in scores.items())
    if not base: base=verdict
    gains=sum(v and not base[i] for i,v in verdict.items())
    losses=sum(not v and base[i] for i,v in verdict.items())
    print(path.parent.relative_to(root),f'{100*macro:.2f}',gains,losses)

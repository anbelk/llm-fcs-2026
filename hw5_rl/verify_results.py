"""Independently recompute saved metrics and check run/data provenance."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):return [json.loads(l)for l in path.read_text().splitlines()if l.strip()]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--require-all',action='store_true');args=ap.parse_args()
    root=Path(__file__).parent; evaluator=root/'materials/evaluate.py'
    module=ast.parse(evaluator.read_text())
    groups=next(ast.literal_eval(n.value)for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='CATEGORY_GROUPS'for t in n.targets))
    benchmark=root/'materials/bfcl.jsonl';ids=[r['id']for r in read(benchmark)]
    val_ids=[(r['dialogue'],r['turn'])for r in read(root/'eda_outputs/validation.jsonl')]
    benchmark_hash=hashlib.sha256(benchmark.read_bytes()).hexdigest()
    results={};configs=[]
    for name in ('baseline','A_strict','B_partial','C_partial_kl'):
        folder=args.runs/name;mp=folder/'bfcl/metrics.json'
        if not mp.exists():
            if args.require_all:raise AssertionError(f'Missing actual results: {name}')
            continue
        metrics=json.loads(mp.read_text());rows=read(folder/'bfcl/predictions.jsonl')
        assert [r['id']for r in rows]==ids and len(ids)==len(set(ids))==350
        flags=defaultdict(list)
        for row in rows:
            flags[row['category']].append(row['check']['valid'])
            if row['finish_reason']=='length':assert not row['check']['valid']
        category={k:mean(v)for k,v in flags.items()}
        weighted=mean(mean(category[k]for k in keys)for keys in groups.values())
        assert abs(metrics['overall']-weighted)<1e-12
        assert metrics['benchmark_sha256']==benchmark_hash
        assert metrics['base_revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
        assert metrics['decoding']=={'temperature':0,'max_new_tokens':1024,'enable_thinking':False,'seed':42}
        vr=read(folder/'validation/predictions.jsonl');vm=json.loads((folder/'validation/metrics.json').read_text())
        assert [(r['dialogue'],r['turn'])for r in vr]==val_ids
        assert len(vr)==vm['n']==1165
        assert abs(mean(r['strict']for r in vr)-vm['strict'])<1e-12
        assert abs(mean(r['partial']for r in vr)-vm['partial'])<1e-12
        assert sum(not r['valid']for r in vr)==vm['invalid']
        if name!='baseline':
            cfg=json.loads((folder/'training/config.json').read_text());configs.append(cfg)
            state=json.loads((folder/'training/trainer_state.json').read_text())
            tm=json.loads((folder/'training/training_metrics.json').read_text())
            assert state['global_step']==cfg['max_steps']
            assert tm['best_checkpoint']==state['best_model_checkpoint']
            assert (folder/'training/grpo_smoke.json').exists()
            env=json.loads((folder/'training/environment.json').read_text())
            for split in ('train','validation'):
                f=root/f'eda_outputs/{split}.jsonl'
                assert env['data_sha256'][f.name]==hashlib.sha256(f.read_bytes()).hexdigest()
        results[name]={'bfcl_predictions':len(rows),'recomputed_overall':weighted,'validation_predictions':len(vr),'checks':'passed'}
    if configs:
        shared=[{k:v for k,v in cfg.items()if k not in ('name','reward','beta')}for cfg in configs]
        assert all(c==shared[0]for c in shared)
    if args.require_all:assert (args.runs/'loss_reward.png').is_file()
    report={'checks':'passed','all_four_present':len(results)==4,'runs':results}
    (args.runs/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

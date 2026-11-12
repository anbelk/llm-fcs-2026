"""Summarize only real files; never synthesize missing run metrics."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);args=ap.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt=None  # GPU image stays unchanged; graphs can be generated on the laptop.
    table=[]
    if plt:fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for name in ('baseline','A_strict','B_partial','C_partial_kl'):
        base=args.runs/name;entry={'run':name,'status':'not_completed'}
        bm=base/'bfcl/metrics.json';vm=base/'validation/metrics.json'
        if bm.exists():
            metrics=json.loads(bm.read_text());entry.update(status='bfcl_completed',overall=metrics['overall'])
            preds=[json.loads(l)for l in (base/'bfcl/predictions.jsonl').read_text().splitlines()if l.strip()]
            if len(preds)!=350:raise ValueError(f'{name}: expected 350 predictions')
        if vm.exists():
            val=json.loads(vm.read_text());entry.update(validation_strict=val['strict'],validation_partial=val['partial'])
        log=base/'training/history.jsonl'
        if log.exists() and plt:
            history=[json.loads(l)for l in log.read_text().splitlines()if l.strip()]
            for ax,key in zip(axes,('loss','reward')):
                points=[(h['step'],h[key])for h in history if key in h]
                if points:ax.plot(*zip(*points),label=name)
        table.append(entry)
    if plt:
        for ax,title in zip(axes,('GRPO loss','Train reward')):
            ax.set_title(title);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=9)
        fig.tight_layout();fig.savefig(args.runs/'loss_reward.png',dpi=160);plt.close(fig)
    keys=['run','status','overall','validation_strict','validation_partial']
    with (args.runs/'comparison.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(table)
    (args.runs/'comparison.json').write_text(json.dumps(table,indent=2)+'\n')
    # Keep individual prediction files intact; inspect errors through their validation_result field.
    print(json.dumps(table,indent=2))
    if not plt:print('Graphs pending: rerun summarize.py with requirements-plot.txt on the laptop.')


if __name__=='__main__':main()

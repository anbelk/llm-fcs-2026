"""Post-evaluation audit. Never changes checkpoint selection or training configs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def read(path):return [json.loads(l)for l in path.read_text().splitlines()if l.strip()]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--benchmark',type=Path,default=Path(__file__).parent/'materials/bfcl.jsonl');args=ap.parse_args()
    benchmark={r['id']:r for r in read(args.benchmark)}
    baseline_path=args.runs/'baseline/bfcl/predictions.jsonl'
    if not baseline_path.exists():raise RuntimeError('No actual baseline predictions')
    baseline={r['id']:r for r in read(baseline_path)}
    analysis={}
    for name in ('baseline','A_strict','B_partial','C_partial_kl'):
        path=args.runs/name/'bfcl/predictions.jsonl'
        if not path.exists():continue
        rows=read(path);failure_categories=Counter();counts=Counter();examples={k:[]for k in ('improved','regressed','still_wrong')}
        by_category={k:{}for k in examples}
        for r in sorted(rows,key=lambda x:x['id']):
            correct=r['check']['valid'];before=baseline[r['id']]['check']['valid']
            kind='improved' if correct and not before else 'regressed' if before and not correct else 'still_wrong' if not correct else 'still_correct'
            counts[kind]+=1
            if not correct:failure_categories[r['category']]+=1
            if kind in examples:
                b=benchmark[r['id']]
                ex={'id':r['id'],'category':r['category'],'question':b['question'],
                    'reference':b['ground_truth'],'baseline_calls':baseline[r['id']]['calls'],
                    'calls':r['calls'],'check':r['check'],'response':r['response'],'finish_reason':r['finish_reason']}
                if len(examples[kind])<2:examples[kind].append(ex)
                by_category[kind].setdefault(r['category'],ex)
        item={'outcomes_vs_baseline':counts,'failures_by_category':failure_categories,
              'examples_first_two_by_id':examples,'first_example_per_category_and_outcome':by_category}
        history=args.runs/name/'training/history.jsonl'
        if history.exists():
            steps=[r for r in read(history) if 'loss'in r and 'frac_reward_zero_std'in r]
            item['training_signal']={'optimizer_steps':len(steps),'prompt_groups':len(steps)*2,
                'zero_advantage_groups':round(sum(r['frac_reward_zero_std']*2 for r in steps)),
                'zero_advantage_group_fraction':mean(r['frac_reward_zero_std']for r in steps),
                'steps_with_all_groups_zero_advantage':sum(r['frac_reward_zero_std']==1 for r in steps),
                'mean_kl':mean(r['kl']for r in steps),
                'note':'Identical rollout rewards within a prompt group give zero relative policy advantage; KL can still update weights.'}
        analysis[name]=item
    (args.runs/'error_analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

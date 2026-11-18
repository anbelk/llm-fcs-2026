import hashlib
import json
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
KINDS = {
    'simple': 'Simple',
    'multiple': 'Multiple',
    'parallel': 'Parallel',
    'parallel_multiple': 'Parallel + Multiple',
    'no_call': 'Без вызова',
}


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def stats(values):
    array = np.asarray(values)
    return dict(zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'),
                    (int(x) for x in np.quantile(array, [0, .5, .9, .95, .99, 1], method='higher'))))


def style(fig):
    fig.patch.set_facecolor('#f8fafc')


def clean(ax, axis='x'):
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.grid(axis=axis, color='#e2e8f0', lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors='#334155')
    ax.set_facecolor('#f8fafc')


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def categories_chart(counts):
    labels = list(KINDS.values())
    train = [counts['train'][kind] for kind in KINDS]
    val = [counts['validation'][kind] for kind in KINDS]
    fig, ax = plt.subplots(figsize=(9.6, 4.25), layout='constrained')
    style(fig)
    ax.barh(labels, train, color='#2455a4', label='Обучение', height=.55)
    ax.barh(labels, val, left=train, color='#80b6e3', label='Валидация', height=.55)
    for i, (a, b) in enumerate(zip(train, val)):
        ax.text(a+b+65, i, f'{a+b:,}'.replace(',', ' '), va='center', color='#172554',
                fontsize=11, weight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.add(train, val))*1.19)
    ax.set_xlabel('Число RL-примеров')
    ax.set_title('ToolACE после раскручивания диалогов', loc='left', weight='bold', pad=15)
    clean(ax)
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'toolace-categories.png')


def length_chart(prompts, answers, tools):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), layout='constrained')
    style(fig)
    for ax, vals, title, limit in zip(
        axes[:2], [prompts, answers],
        ['Готовый промпт', 'Ответ ассистента в ToolACE'], [3072, 512],
    ):
        vals = np.sort(np.asarray(vals))
        ax.step(vals, 100*np.arange(1, len(vals)+1)/len(vals), where='post', color='#2455a4', lw=2)
        ax.axvline(limit, color='#b45309', lw=1.1, ls='--', label=f'Рабочий лимит {limit}')
        ax.set_xscale('log')
        ticks = [300, 600, 1000, 2000, 4000] if limit == 3072 else [2, 10, 50, 200, 1000]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0,102)
        ax.set_xlabel('Токены (лог. шкала)')
        ax.set_ylabel('Примеров не длиннее, %')
        ax.set_title(title, loc='left', fontsize=11, weight='bold')
        clean(ax, 'y')
        ax.legend(frameon=False, loc='lower right', fontsize=8)
    ax = axes[2]
    c = Counter(tools)
    xs = sorted(c)
    ax.bar(xs, [c[x] for x in xs], color='#2455a4', width=.73)
    ax.set_xticks(xs)
    ax.set_xlabel('Доступных инструментов')
    ax.set_ylabel('RL-примеров')
    ax.set_title('Инструменты на пример', loc='left', fontsize=11, weight='bold')
    clean(ax, 'y')
    save(fig, 'toolace-lengths-and-tools.png')


def main(source: Path, tokenizer_dir: Path | None = None):
    raw_bytes = source.read_bytes()
    raw = json.loads(raw_bytes)

    def model_file(name):
        if tokenizer_dir is not None:
            return tokenizer_dir / name
        return Path(hf_hub_download(MODEL, name, revision=REVISION))
    splits = {s: read_jsonl(ROOT / 'data' / f'{s}.jsonl') for s in ('train', 'validation')}
    all_rows = splits['train'] + splits['validation']
    counts = {s: Counter(r['category'] for r in rows) for s, rows in splits.items()}
    tok = Tokenizer.from_file(str(model_file('tokenizer.json')))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)

    def raise_exception(message):
        raise ValueError(message)
    env.globals['raise_exception'] = raise_exception
    template = env.from_string(model_file('chat_template.jinja').read_text())
    lengths = []
    for start in range(0, len(all_rows), 128):
        batch = all_rows[start:start+128]
        rendered = [template.render(messages=r['prompt'], tools=r['tools'],
                                    add_generation_prompt=True, enable_thinking=False)
                    for r in batch]
        lengths.extend(len(v.ids) for v in tok.encode_batch(rendered, add_special_tokens=False))
    targets = [raw[r['source_row']]['conversations'][r['turn']]['value'] for r in all_rows]
    answer_lengths = [len(v.ids) for v in tok.encode_batch(targets, add_special_tokens=False)]
    tool_counts = [len(r['tools']) for r in all_rows]
    by_category = {
        k: {'prompt_tokens': stats([x for x,r in zip(lengths,all_rows) if r['category']==k]),
            'reference_tokens': stats([x for x,r in zip(answer_lengths,all_rows) if r['category']==k]),
            'prompts_above_3072': sum(x > 3072 for x,r in zip(lengths,all_rows) if r['category']==k)}
        for k in KINDS
    }
    OUT.mkdir(exist_ok=True)
    categories_chart(counts)
    length_chart(lengths, answer_lengths, tool_counts)
    result = {
        'source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'raw_dialogues': len(raw),
        'raw_dialogue_messages': stats([len(d['conversations']) for d in raw]),
        'raw_message_roles': dict(Counter(m['from'] for r in raw for m in r['conversations'])),
        'raw_single_user_turn_dialogues': sum(sum(m['from']=='user' for m in r['conversations'])==1 for r in raw),
        'prepared_examples': {s: len(rows) for s,rows in splits.items()},
        'category_counts': counts,
        'prompt_tokens': stats(lengths),
        'source_reference_tokens': stats(answer_lengths),
        'tools_per_example': stats(tool_counts),
        'most_common_tool_counts': Counter(tool_counts).most_common(5),
        'unique_normalized_tool_names': len({t['function']['name'] for r in all_rows for t in r['tools']}),
        'prompts_above_3072': sum(x > 3072 for x in lengths),
        'source_references_above_512': sum(x > 512 for x in answer_lengths),
        'by_category': by_category,
    }
    (OUT / 'toolace-eda.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_category','source_sha256')}, indent=2))


if __name__ == '__main__':
    typer.run(main)

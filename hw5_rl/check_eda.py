"""Parser edge cases and conservation/leakage checks for saved EDA exports."""
import argparse
import hashlib
import json
from pathlib import Path

from eda import parse_calls, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--compare', type=Path)
    ap.add_argument('--check-token-lengths', action='store_true')
    args = ap.parse_args()
    text = '''[Get Agent's Active Listings(zuid="X"), Search for Words in Title, Text, or URL(query="comma, ) ]", nested={"x":[1,true,null]}), User Feed (Video Posts) V2(from="x=y")]'''
    names = ["Get Agent's Active Listings", 'Search for Words in Title, Text, or URL', 'User Feed (Video Posts) V2']
    parsed = parse_calls(text, names)
    assert len(parsed) == 3 and parsed[1]['arguments']['nested']['x'] == [1, True, None]
    assert parsed[2]['arguments'] == {'from':'x=y'}
    assert parse_calls('Please specify a city.', ['f']) is None
    assert parse_calls('[]', ['f']) == []
    for text in ['[f(x=1,x=2)]','[f(x=__import__("os"))]', '[unknown()]',
                 '[f(x=[1,2)]', '[f(x=NaN)]', '[f(x=1),]']:
        try:
            parse_calls(text, ['f'])
        except (ValueError, SyntaxError):
            continue
        raise AssertionError(f'Accepted invalid call: {text}')
    assert validate({'x':'1'}, {'type':'dict','properties':{'x':{'type':'int'}}}) == [('type_mismatch','.x')]
    assert validate({}, {'type':'dict','properties':{'x':{'type':'string'}},'required':['x']}) == [('missing_required','.x')]
    # A free-form object is allowed to contain arbitrary keys.
    assert not validate({'x':1}, {'type':'dict'})
    stats = json.loads((args.out/'stats.json').read_text())
    c = stats['counts']
    assert c['all_assistant_turns'] == stats['roles']['assistant'] == 13819
    assert c['all_decisions_after_user'] == c['after_user'] + 500 == 12452
    assert sum(stats['available_tools_per_dialogue'].values()) == c['dialogues'] - 500
    assert sum(stats['calls_per_assistant_turn'].values()) == c['assistant_turns']
    assert sum(stats['arguments_per_call'].values()) == c['calls']
    assert sum(int(k)*v for k,v in stats['arguments_per_call'].items()) == c['arguments']
    assert c['after_user'] == c['candidate_decisions'] + c['excluded_bad_current_target']
    assert c['candidate_decisions'] == c['selected_decisions'] + c['excluded_duplicate_prompts'] + c['excluded_length']
    rows = {s:[json.loads(l) for l in (args.out/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation')}
    manifest = json.loads((args.out/'split_manifest.json').read_text())['dialogue_split']
    for s, rs in rows.items():
        assert len(rs) == stats['split'][s]['selected_decisions']
        assert all(manifest[str(r['dialogue'])] == s == r['split'] for r in rs)
        assert all(r['prompt_tokens'] <= 3072 and r['target_tokens'] <= 1024 for r in rs)
        assert all(r['prompt'][-1]['role'] == 'user' for r in rs)
    for key in ('dialogue','prompt_hash'):
        assert not ({r[key] for r in rows['train']} & {r[key] for r in rows['validation']})
    if args.check_token_lengths:
        from transformers import AutoTokenizer
        from eda import MODEL_REVISION
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision=MODEL_REVISION)
        for rs in rows.values():
            for r in rs:
                tokens = tokenizer.apply_chat_template(r['prompt'], tools=r['tools'], tokenize=True,
                         return_dict=False, add_generation_prompt=True, enable_thinking=False)
                assert len(tokens) == r['prompt_tokens'], (r['dialogue'], r['turn'])
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()}
    if args.compare:
        other = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.compare.iterdir()) if p.is_file()}
        assert hashes == other, 'Rerun differs byte-for-byte'
    print(json.dumps({'checks':'passed','byte_identical_rerun':bool(args.compare),
                      'exported_prompt_token_lengths_checked':args.check_token_lengths,'sha256':hashes},indent=2))


if __name__ == '__main__':
    main()

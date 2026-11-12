"""CPU-only full reference round trip through the real template and native parser."""
import argparse
from collections import Counter
import json
from pathlib import Path

from eda import MODEL_REVISION
from rewards import NativeParser,score_text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=MODEL_REVISION)
    parser=NativeParser(tokenizer);counts=Counter();exceptions=[]
    for split in ('train','validation'):
        for line in (Path(__file__).parent/f'eda_outputs/{split}.jsonl').read_text().splitlines():
            r=json.loads(line);calls=r['reference']['calls']
            if not calls:counts[split+'_no_call']+=1;continue
            counts[split+'_call_targets']+=1
            prefix=tokenizer.apply_chat_template(r['prompt'],tools=r['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            full=tokenizer.apply_chat_template(r['prompt']+[{'role':'assistant','content':'','tool_calls':[{'type':'function','function':c}for c in calls]}],tools=r['tools'],tokenize=False,enable_thinking=False)
            assert full.startswith(prefix)
            value,info=score_text(full[len(prefix):],r['tools'],r['reference'],parser,'strict',numeric_equivalence=True)
            counts[split+'_roundtrip_passed']+=value==1
            if value!=1:exceptions.append({'split':split,'dialogue':r['dialogue'],'turn':r['turn'],'reference':r['reference'],'parsed':info})
    result={'counts':counts,'exceptions':exceptions,'data_changed':False}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(counts))


if __name__=='__main__':main()

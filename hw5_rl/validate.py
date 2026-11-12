"""Deterministic ToolACE validation only. Never loads BFCL."""
import argparse
import json
from pathlib import Path
from statistics import mean

from eda import MODEL_REVISION
from rewards import NativeParser,score_text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    ap.add_argument('--data',type=Path,default=Path(__file__).parent/'eda_outputs/validation.jsonl')
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=MODEL_REVISION)
    rows=[json.loads(s)for s in args.data.read_text().splitlines()if s.strip()]
    prompts=[{'prompt_token_ids':tokenizer.apply_chat_template(r['prompt'],tools=r['tools'],tokenize=True,return_dict=False,
              add_generation_prompt=True,enable_thinking=False)}for r in rows]
    llm=LLM(model=args.model,revision=MODEL_REVISION if args.model=='Qwen/Qwen3.5-2B' else None,
            tokenizer='Qwen/Qwen3.5-2B',tokenizer_revision=MODEL_REVISION,generation_config='vllm',
            dtype='bfloat16',language_model_only=True,max_model_len=4096,max_num_seqs=64,
            gpu_memory_utilization=.85,enforce_eager=True,seed=42)
    generated=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1024,skip_special_tokens=False))
    parser=NativeParser(tokenizer);results=[]
    for r,g in zip(rows,generated,strict=True):
        o=g.outputs[0]
        strict,info=score_text(o.text,r['tools'],r['reference'],parser,'strict',o.finish_reason=='length',True)
        partial,_=score_text(o.text,r['tools'],r['reference'],parser,'partial',o.finish_reason=='length',True)
        results.append({'dialogue':r['dialogue'],'turn':r['turn'],'response':o.text,'finish_reason':o.finish_reason,
                        'strict':strict,'partial':partial,**info})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in results))
    (args.output/'metrics.json').write_text(json.dumps({'n':len(results),'strict':mean(r['strict']for r in results),
        'partial':mean(r['partial']for r in results),'invalid':sum(not r['valid']for r in results)},indent=2)+'\n')


if __name__=='__main__':main()

import argparse
import json
from pathlib import Path

from eda import MODEL_REVISION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Refusing to overwrite merged weights')
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    base=snapshot_download('Qwen/Qwen3.5-2B',revision=MODEL_REVISION)
    model=AutoModelForImageTextToText.from_pretrained(base,dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,args.adapter).merge_and_unload(safe_merge=True)
    model.config.use_cache=True
    model.save_pretrained(args.output,safe_serialization=True,max_shard_size='4GB')
    # Evaluation always uses the original tokenizer by fixed ID/revision; never save a modified tokenizer.
    (args.output/'merge_provenance.json').write_text(json.dumps({'base':'Qwen/Qwen3.5-2B','revision':MODEL_REVISION,'adapter':str(Path(args.adapter).resolve())},indent=2)+'\n')


if __name__=='__main__':main()

"""Single-GPU GRPO/LoRA. Use inside the assignment's pinned Docker image."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

from eda import MODEL_REVISION, digest
from rewards import make_reward

BASE='Qwen/Qwen3.5-2B'
VERSIONS={'transformers':'5.9.0','trl':'1.16.0','peft':'0.18.1','vllm':'0.22.0','datasets':'4.7.0','accelerate':'1.10.1','numpy':'2.2.6','bfcl-eval':'2026.3.23'}


def check_versions():
    actual={k:version(k) for k in VERSIONS}
    if {k:v.split('+')[0] for k,v in actual.items()}!=VERSIONS:
        raise RuntimeError(f'Expected {VERSIONS}, found {actual}')
    return actual


def load_rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dataset(rows,tokenizer):
    from datasets import Dataset
    records=[]
    for r in rows:
        prompt=tokenizer.apply_chat_template(r['prompt'],tools=r['tools'],tokenize=False,
                                            add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)!=r['prompt_tokens'] or len(ids)>3072: raise ValueError('Prompt changed since EDA')
        # Plain strings are already rendered; TRL must not template them a second time.
        records.append({'prompt':prompt,'reference_json':json.dumps(r['reference'],ensure_ascii=False),
                        'tools_json':json.dumps(r['tools'],ensure_ascii=False),
                        'dialogue':r['dialogue'],'turn':r['turn']})
    return Dataset.from_list(records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,default=Path(__file__).parent/'eda_outputs')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--base-path',default=BASE,help='Official pinned snapshot, or Hub ID; never another model')
    args=ap.parse_args(); cfg=json.loads(args.config.read_text())
    versions=check_versions()
    import torch
    from transformers import AutoModelForImageTextToText,AutoTokenizer,TrainerCallback,set_seed
    from huggingface_hub import snapshot_download
    from trl import GRPOConfig,GRPOTrainer
    from peft import LoraConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 GPU required')
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError('Output exists; refusing to overwrite a run')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    set_seed(cfg['seed'])
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,padding_side='left')
    train_rows=load_rows(args.data_dir/'train.jsonl'); val_rows=load_rows(args.data_dir/'validation.jsonl')
    # Fixed subset chosen by source IDs, independent of reference labels and BFCL.
    val_rows=sorted(val_rows,key=lambda r:digest(['validation-selection-42',r['dialogue'],r['turn']]))[:cfg['validation_selection_size']]
    train_data=dataset(train_rows,tokenizer); val_data=dataset(val_rows,tokenizer)
    pinned_path=snapshot_download(BASE,revision=MODEL_REVISION)
    base_path=pinned_path if args.base_path==BASE else args.base_path
    if Path(base_path).resolve()!=Path(pinned_path).resolve():raise RuntimeError('Use the official pinned snapshot returned by snapshot_download')
    base_config=json.loads((Path(base_path)/'config.json').read_text())
    if base_config.get('model_type')!='qwen3_5': raise RuntimeError('Expected the original Qwen3.5 architecture')
    model=AutoModelForImageTextToText.from_pretrained(base_path,
                                            dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    # Keep the original architecture/weight names for native vLLM loading and merging.
    # Vision weights remain frozen; LoRA is applied only to the language model.
    targets=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and name.startswith('model.language_model.')]
    if not targets: raise RuntimeError('Language-model LoRA targets not found')
    trainable_cfg=LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=0.,
                            target_modules=targets,bias='none',task_type='CAUSAL_LM')
    primary=make_reward(tokenizer,cfg['reward'],1024,cfg['numeric_equivalence'],model.generation_config.eos_token_id)
    common=make_reward(tokenizer,'strict',1024,cfg['numeric_equivalence'],model.generation_config.eos_token_id); common.__name__='common_strict'
    params=GRPOConfig(output_dir=str(args.output),seed=cfg['seed'],data_seed=cfg['seed'],
        learning_rate=cfg['learning_rate'],max_steps=cfg['max_steps'],
        per_device_train_batch_size=1,gradient_accumulation_steps=8,per_device_eval_batch_size=4,
        num_generations=4,num_generations_eval=4,max_completion_length=1024,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        beta=cfg['beta'],temperature=1.,top_p=1.,top_k=0,loss_type='grpo',scale_rewards='group',
        reward_weights=[1.,0.],mask_truncated_completions=True,
        use_vllm=cfg['use_vllm'],vllm_mode='colocate',vllm_gpu_memory_utilization=.35,
        vllm_max_model_length=4096,vllm_enable_sleep_mode=True,
        eval_strategy='steps',eval_steps=cfg['checkpoint_steps'],save_strategy='steps',
        save_steps=cfg['checkpoint_steps'],save_total_limit=2,load_best_model_at_end=True,
        metric_for_best_model='eval_rewards/common_strict/mean',greater_is_better=True,
        logging_steps=1,report_to='none',remove_unused_columns=False,
        optim='adamw_torch',lr_scheduler_type='constant',warmup_steps=0,
        disable_dropout=True,log_completions=False)
    class Log(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if state.is_world_process_zero:
                if logs and any(not math.isfinite(float(logs[k])) for k in ('loss','grad_norm','reward') if k in logs):
                    raise RuntimeError('Non-finite GRPO training metric')
                with (Path(args.output_dir)/'history.jsonl').open('a') as f:
                    f.write(json.dumps({'step':state.global_step,**(logs or {})})+'\n')
                if state.global_step==1 and logs and 'loss'in logs:
                    (Path(args.output_dir)/'grpo_smoke.json').write_text(json.dumps({'status':'passed','step':1,
                        'checks':['native rollout','reward evaluation','backpropagation and optimizer step','finite loss/gradient/reward'],
                        'logs':logs},indent=2)+'\n')
    meta={'versions':{**versions,'torch':torch.__version__},'base':BASE,'revision':MODEL_REVISION,
          'train_rows':len(train_rows),'validation_selection_ids':[[r['dialogue'],r['turn']]for r in val_rows],
          'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [args.data_dir/'train.jsonl',args.data_dir/'validation.jsonl']},
          'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'lora_targets':targets,
          'numeric_comparison':'typed' if not cfg['numeric_equivalence'] else 'int_float_equivalent'}
    (args.output/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
    trainer=GRPOTrainer(model=model,args=params,train_dataset=train_data,eval_dataset=val_data,
                        processing_class=tokenizer,peft_config=trainable_cfg,reward_funcs=[primary,common],callbacks=[Log()])
    start=time.monotonic(); result=trainer.train()
    trainer.save_model(str(args.output/'adapter'))
    trainer.state.save_to_json(str(args.output/'trainer_state.json'))
    metrics={**result.metrics,'elapsed_seconds':time.monotonic()-start,
             'best_checkpoint':trainer.state.best_model_checkpoint,'best_metric':trainer.state.best_metric,
             'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    (args.output/'training_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':main()

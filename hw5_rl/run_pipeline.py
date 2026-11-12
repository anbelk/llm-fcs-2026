"""Baseline + three fixed RL configs. Sequential GPU processes free VRAM per stage.

Run from the mounted repository in the assignment image. No evaluator edits.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now():return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parent; runs=args.runs.resolve();runs.mkdir(parents=True,exist_ok=True)
    manifest_path=runs/'run_status.json'
    if manifest_path.exists() and not args.resume:raise RuntimeError('Run folder exists; use --resume for completed stages, or a new folder')
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'created_utc':now(),'stages':{}}
    def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    expected=json.loads((root/'provenance.json').read_text())
    for name,item in expected['materials'].items():
        assert hashlib.sha256((root/'materials'/name).read_bytes()).hexdigest()==item['sha256'],name
    def stage(name,cmd,cwd=root):
        old=manifest['stages'].get(name,{})
        if old.get('status')=='completed':return
        log=runs/(name.replace('/','_')+'.log')
        manifest['stages'][name]={'status':'running','started_utc':now(),'command':list(map(str,cmd)),'cwd':str(cwd),'log':str(log)};save()
        with log.open('a')as f:
            result=subprocess.run(list(map(str,cmd)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        manifest['stages'][name].update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished_utc=now());save()
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
    python=sys.executable
    stage('unit_tests',[python,'-m','unittest','discover','-s',root,'-p','test_rewards.py','-v'])
    stage('native_parser',[python,root/'native_smoke.py'])
    from huggingface_hub import snapshot_download
    from eda import MODEL_REVISION
    from train import check_versions
    versions=check_versions();manifest['versions']=versions;save()
    base=snapshot_download('Qwen/Qwen3.5-2B',revision=MODEL_REVISION)
    stage('baseline/bfcl',[python,root/'materials/evaluate.py',base,'--output',runs/'baseline/bfcl'],root/'materials')
    stage('baseline/validation',[python,root/'validate.py','--model',base,'--output',runs/'baseline/validation'])
    for name in ('A_strict','B_partial','C_partial_kl'):
        out=runs/name
        stage(name+'/train',[python,root/'train.py','--config',root/f'configs/{name}.json','--output',out/'training','--base-path',base])
        stage(name+'/merge',[python,root/'merge.py','--adapter',out/'training/adapter','--output',out/'merged'])
        stage(name+'/validation',[python,root/'validate.py','--model',out/'merged','--output',out/'validation'])
        stage(name+'/bfcl',[python,root/'materials/evaluate.py',out/'merged','--output',out/'bfcl'],root/'materials')
    stage('summary',[python,root/'summarize.py','--runs',runs])
    manifest['completed_utc']=now();save()


if __name__=='__main__':main()

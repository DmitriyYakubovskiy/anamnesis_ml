import argparse
from common.data import ROOT, prepare, export_task_formats

p=argparse.ArgumentParser()
p.add_argument('--input',default=str(ROOT/'data/raw/anamneses.json'))
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
for folder in ['Lora','ner_classifier','qwen']:
    dest=ROOT/folder/'data'
    manifest,parts=prepare(a.input,dest,a.seed)
    export_task_formats(folder,dest,manifest,parts)
    print(folder,{s:len(rr) for s,rr in parts.items()})

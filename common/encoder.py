"""Separate BIO recognizer and three-/multi-class history classifier for each active column."""
import argparse
from pathlib import Path
import time
from common.data import ROOT, load, read_jsonl, dump, jsonl, key
from common.metrics import value_metrics, span_metrics, save_metrics
from common.runtime import DEFAULT_ENCODER, new_output, training_args, save_training, versions


def tokenize_checked(tokenizer, text, max_length):
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=False)
    if len(encoded['input_ids']) > max_length:
        raise ValueError(f'Text needs {len(encoded["input_ids"])} tokens; limit {max_length}. Use a longer-context encoder. No silent truncation.')
    return encoded


def bio_labels(offsets, spans):
    labels = [0 if e > s else -100 for s,e in offsets]
    expanded = []
    for span in sorted(spans,key=lambda a:a['start']):
        indices = [i for i,(s,e) in enumerate(offsets) if e>s and s<span['end'] and e>span['start']]
        if not indices:
            raise ValueError('Gold span contains no tokenizable characters')
        if any(labels[i] != 0 for i in indices):
            raise ValueError('Two spans share a token; BIO cannot represent this markup')
        for j,i in enumerate(indices):
            labels[i] = 1 if j==0 else 2
        if (offsets[indices[0]][0], offsets[indices[-1]][1]) != (span['start'],span['end']):
            expanded.append({'gold':[span['start'],span['end']], 'token_bounds':[offsets[indices[0]][0],offsets[indices[-1]][1]]})
    return labels, expanded


def decode_bio(ids, offsets):
    result, current = [], None
    for tag,(s,e) in zip(ids,offsets):
        if e <= s:
            continue
        if tag==1 or (tag==2 and current is None):
            if current:
                result.append(tuple(current))
            current=[s,e]
        elif tag==2:
            current[1]=e
        elif current:
            result.append(tuple(current));current=None
    if current:
        result.append(tuple(current))
    return result


def train():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(ROOT/'ner_classifier/data'))
    p.add_argument('--model',default=DEFAULT_ENCODER)
    p.add_argument('--revision',default='main')
    p.add_argument('--output',default=str(ROOT/'runs/encoder'))
    p.add_argument('--task',choices=['both','ner','classifier'],default='both')
    p.add_argument('--epochs',type=float,default=5)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--max-length',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=-1)
    a=p.parse_args()
    from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification, DataCollatorForTokenClassification, DataCollatorWithPadding, Trainer, set_seed
    data=Path(a.data);schema=load(data/'schema.json');parts={s:read_jsonl(data/f'{s}.jsonl') for s in ['train','validation']}
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('NER needs a fast tokenizer with offsets')
    base_config=AutoConfig.from_pretrained(a.model,revision=a.revision)
    if a.max_length>getattr(base_config,'max_position_embeddings',a.max_length):
        p.error('--max-length exceeds encoder position embeddings')
    out=new_output(a.output)
    tasks=['ner','classifier'] if a.task=='both' else [a.task]
    for col in schema:
        code=col['code']
        for task in tasks:
            set_seed(a.seed)
            label_names=['O','B-'+code,'I-'+code] if task=='ner' else [key(v) for v in col['values']]
            label2id={v:i for i,v in enumerate(label_names)}
            datasets,alignment={},[]
            for s,rows in parts.items():
                examples=[]
                for r in rows:
                    enc=tokenize_checked(tokenizer,r['text'],a.max_length)
                    offsets=enc.pop('offset_mapping')
                    if task=='ner':
                        labels,expanded=bio_labels(offsets,[x for x in r['spans'] if x['label']==code])
                        alignment.extend({'id':r['id'],'split':s,**x} for x in expanded)
                    else:
                        labels=label2id[key(r['target'][code])]
                    examples.append({**enc,'labels':labels})
                datasets[s]=examples
            cls=AutoModelForTokenClassification if task=='ner' else AutoModelForSequenceClassification
            model=cls.from_pretrained(a.model,revision=a.revision,num_labels=len(label_names),id2label=dict(enumerate(label_names)),label2id=label2id,ignore_mismatched_sizes=True)
            dest=out/code/task;dest.mkdir(parents=True)
            collator=DataCollatorForTokenClassification(tokenizer) if task=='ner' else DataCollatorWithPadding(tokenizer)
            trainer=Trainer(model=model,args=training_args(dest,a.epochs,a.lr,a.batch,1,a.seed,a.max_steps),
                train_dataset=datasets['train'],eval_dataset=datasets['validation'],data_collator=collator,processing_class=tokenizer)
            trainer.train()
            config={**vars(a),'task':task,'column':col,'base_commit':getattr(model.config,'_commit_hash',None),'source_sha256':load(data/'manifest.json')['source_sha256']}
            save_training(trainer,tokenizer,dest,config)
            dump(dest/'alignment_report.json',alignment)
    dump(out/'run_config.json',vars(a))
    dump(out/'schema.json',schema)


def evaluate():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(ROOT/'ner_classifier/data'))
    p.add_argument('--models',default=str(ROOT/'runs/encoder'))
    p.add_argument('--output',default=str(ROOT/'runs/encoder_eval'))
    p.add_argument('--task',choices=['both','ner','classifier'],default='both')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--max-length',type=int,default=2048)
    a=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoModelForSequenceClassification
    data=Path(a.data);schema=load(data/'schema.json');rows=read_jsonl(data/f'{a.split}.jsonl');out=new_output(a.output)
    if load(Path(a.models)/'schema.json') != schema:
        raise ValueError('Trained model schema differs from evaluation data')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictions={r['id']:{'id':r['id'],'prediction':{},'spans':[],'errors':[]} for r in rows}
    gold_spans,pred_spans=[],[];token_gold,token_pred=[],[];alignment=[]
    started=time.perf_counter()
    tasks=['ner','classifier'] if a.task=='both' else [a.task]
    for col in schema:
        code=col['code']
        for task in tasks:
            path=Path(a.models)/code/task/'model'
            tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True)
            cls=AutoModelForTokenClassification if task=='ner' else AutoModelForSequenceClassification
            model=cls.from_pretrained(path).to(device).eval()
            for r in rows:
                if task=='ner':
                    gold_spans.extend((r['id'],code,x['start'],x['end']) for x in r['spans'] if x['label']==code)
                try:
                    enc=tokenize_checked(tokenizer,r['text'],min(a.max_length,model.config.max_position_embeddings))
                    offsets=enc.pop('offset_mapping')
                    inputs={k:torch.tensor([v],device=device) for k,v in enc.items()}
                    with torch.inference_mode():
                        logits=model(**inputs).logits[0]
                    if task=='classifier':
                        idx=int(logits.argmax())
                        predictions[r['id']]['prediction'][code]=__import__('json').loads(model.config.id2label[idx])
                    else:
                        ids=logits.argmax(-1).cpu().tolist()
                        gold,expanded=bio_labels(offsets,[x for x in r['spans'] if x['label']==code])
                        alignment.extend({'id':r['id'],**x} for x in expanded)
                        token_gold.extend(g for g in gold if g!=-100)
                        token_pred.extend(v for g,v in zip(gold,ids) if g!=-100)
                        decoded=decode_bio(ids,offsets)
                        pred_spans.extend((r['id'],code,s,e) for s,e in decoded)
                        predictions[r['id']]['spans'].extend({'label':code,'start':s,'end':e,'text':r['text'][s:e]} for s,e in decoded)
                except Exception as e:
                    predictions[r['id']]['errors'].append({'column':code,'task':task,'error':f'{type(e).__name__}: {e}'})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    items=list(predictions.values());jsonl(out/'predictions.jsonl',items)
    metrics=value_metrics(rows,items,schema) if 'classifier' in tasks else {}
    if 'ner' in tasks:
        metrics['ner_exact_spans']=span_metrics(gold_spans,pred_spans)
        metrics['ner_token_accuracy']=sum(g==p for g,p in zip(token_gold,token_pred))/len(token_gold) if token_gold else 0
        metrics['ner_entity_token_detection']=span_metrics([i for i,g in enumerate(token_gold) if g>0],[i for i,pred in enumerate(token_pred) if pred>0])
        metrics['ner_evaluated_tokens']=len(token_gold)
        metrics['ner_unrepresentable_boundaries']=len(alignment)
    metrics['record_errors']=sum(bool(r['errors']) for r in items)
    save_metrics(out,metrics,{**vars(a),'versions':versions(),'seconds':time.perf_counter()-started,'manifest':load(data/'manifest.json')})
    dump(out/'errors.json',[r for r in items if r['errors']])
    dump(out/'alignment_report.json',alignment)

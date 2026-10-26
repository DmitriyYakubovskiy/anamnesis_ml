"""Small-data CPU alternative inspired by the article: span NER + TF-IDF classification.

Only train is fitted. Validation chooses epochs/candidates. No medical rules or
validation spans are injected into model predictions; original offsets stay intact.
"""
from collections import Counter
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import random
import time

from common.data import load, read_jsonl, dump, jsonl, key
from common.encoder_training import audit_training_parts
from common.metrics import classification_report, value_metrics, span_metrics, save_metrics
from common.runtime import new_output, versions


def ner_report(rows, predictions, code):
    gold, predicted, gold_chars, pred_chars = [], [], set(), set()
    by_id = {p['id']: p for p in predictions}
    for row in rows:
        for span in row['spans']:
            if span['label']==code:
                gold.append((row['id'],code,span['start'],span['end']))
                gold_chars.update((row['id'],i) for i in range(span['start'],span['end']))
        for span in by_id.get(row['id'],{}).get('spans',[]):
            if span['label']==code:
                predicted.append((row['id'],code,span['start'],span['end']))
                pred_chars.update((row['id'],i) for i in range(span['start'],span['end']))
    exact, chars = span_metrics(gold,predicted), span_metrics(gold_chars,pred_chars)
    return {'n':len(rows),'ner_exact_spans':exact,'ner_entity_characters':chars,
            'ner_f1':exact['f1'],'ner_precision':exact['precision'],'ner_recall':exact['recall'],
            'no_predicted_entities':bool(gold) and not predicted}


def ner_selection(report, loss=0):
    # Genuine lexicographic order: a better exact F1 can never lose to a tie-break.
    return (report['ner_f1'],report['ner_entity_characters']['f1'],
            -report['ner_exact_spans']['fp'],-loss)


def spacy_predictions(nlp, rows, code):
    return [{'id':r['id'],'spans':[{'label':code,'start':e.start_char,'end':e.end_char,'text':e.text}
             for e in doc.ents if e.label_==code]} for r,doc in zip(rows,nlp.pipe([r['text'] for r in rows]))]


def make_examples(nlp, rows, code):
    from spacy.training import Example
    examples, alignment = [], []
    for row in rows:
        doc = nlp.make_doc(row['text'])
        entities = []
        for span in sorted(row['spans'],key=lambda a:(a['start'],a['end'])):
            if span['label']!=code:
                continue
            entity=doc.char_span(span['start'],span['end'],label=code,alignment_mode='expand')
            if entity is None:
                raise ValueError(f"Unrepresentable spaCy span in {row['id']}")
            if entities and entity.start_char < entities[-1][1]:
                raise ValueError('Expanded gold spans overlap; fix annotation/tokenizer before training')
            entities.append((entity.start_char,entity.end_char,code))
            if (entity.start_char,entity.end_char)!=(span['start'],span['end']):
                alignment.append({'id':row['id'],'gold':[span['start'],span['end']],
                                  'training_bounds':[entity.start_char,entity.end_char]})
        examples.append(Example.from_dict(doc,{'entities':entities}))
    return examples,alignment


def train_spacy(rows, validation, column, dest, epochs, seed):
    import spacy
    from spacy.util import fix_random_seed,minibatch,compounding
    fix_random_seed(seed)
    nlp=spacy.blank('ru');ner=nlp.add_pipe('ner');ner.add_label(column['code'])
    examples,alignment=make_examples(nlp,rows,column['code'])
    if not any(a['label']==column['code'] for r in rows for a in r['spans']):
        raise ValueError('Cannot train NER without positive train spans')
    optimizer=nlp.initialize(get_examples=lambda: examples)
    optimizer.learn_rate=.001
    optimizer.grad_clip=1.0
    best=None;history=[];rng=random.Random(seed);sizes=compounding(1.,8.,1.01)
    for epoch in range(1,epochs+1):
        shuffled=examples[:];rng.shuffle(shuffled);losses={}
        dropout=.3-.2*(epoch-1)/max(1,epochs-1)
        for batch in minibatch(shuffled,size=sizes):
            nlp.update(batch,sgd=optimizer,drop=dropout,losses=losses)
        with nlp.use_params(optimizer.averages) if optimizer.averages else nullcontext():
            predictions=spacy_predictions(nlp,validation,column['code'])
            report=ner_report(validation,predictions,column['code'])
            score=ner_selection(report,losses.get('ner',0))
            entry={'epoch':epoch,'train_loss':float(losses.get('ner',0)),'dropout':dropout,
                   'validation':report}
            history.append(entry)
            if best is None or score>best:
                best=score;best_epoch=epoch
                nlp.to_disk(dest/'model')
        print(f"spaCy {column['code']} epoch={epoch}/{epochs} val_exact_f1={report['ner_f1']:.4f} char_f1={report['ner_entity_characters']['f1']:.4f}",flush=True)
        dump(dest/'training_metrics.json',{'backend':'spacy','selection_split':'validation','best_epoch':best_epoch,
             'selection_order':['exact_span_f1','entity_character_f1','fewer_false_positives','lower_train_loss'],
             'history':history,'seed':seed})
    nlp=spacy.load(dest/'model')
    dump(dest/'alignment_report.json',alignment)
    for split,rr in [('train_diagnostic',rows),('validation_best',validation)]:
        pp=spacy_predictions(nlp,rr,column['code'])
        report=ner_report(rr,pp,column['code'])
        dump(dest/f'{split}_metrics.json',{'split':'train' if split=='train_diagnostic' else 'validation',
             'diagnostic_only':split=='train_diagnostic','best_epoch':best_epoch,'metrics':report})
        jsonl(dest/f'{split}_predictions.jsonl',pp)
    return {'backend':'spacy','best_epoch':best_epoch,'validation':ner_report(validation,spacy_predictions(nlp,validation,column['code']),column['code'])}


def classifier_pipeline(C=1.,algorithm='svm',seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    # Keep negation words. Character n-grams absorb endings/abbreviations without
    # changing source text/offsets or requiring a medical spell-correction model.
    features=FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',sublinear_tf=True)),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True)),
    ])
    estimator=LinearSVC(C=C,class_weight='balanced',random_state=seed,dual='auto',max_iter=5000) if algorithm=='svm' else LogisticRegression(C=C,class_weight='balanced',random_state=seed,max_iter=3000)
    return Pipeline([('tfidf',features),('classifier',estimator)])


def train_classifier(rows, validation, column, dest, seed):
    import joblib
    from sklearn.dummy import DummyClassifier
    classes=[key(v) for v in column['values']]
    train_y=[key(r['target'][column['code']]) for r in rows]
    val_y=[key(r['target'][column['code']]) for r in validation]
    train_x=[r['text'] for r in rows];val_x=[r['text'] for r in validation]
    best=None;history=[];winner=None
    grid=[('svm',.25),('svm',1.),('svm',4.),('logistic',1.)] if len(set(train_y))>1 else [('dummy',1.)]
    for algorithm,C in grid:
        pipeline=classifier_pipeline(C,algorithm,seed) if algorithm!='dummy' else DummyClassifier(strategy='most_frequent')
        pipeline.fit(train_x,train_y)
        predictions=pipeline.predict(val_x).tolist()
        report=classification_report(val_y,predictions,classes)
        history.append({'algorithm':algorithm,'C':C,'validation':report})
        score=(report['macro_f1'],report['accuracy'])
        if best is None or score>best:
            best=score;winner=pipeline;selected={'algorithm':algorithm,'C':C}
    joblib.dump(winner,dest/'model.joblib')
    dump(dest/'training_metrics.json',{'backend':'tfidf','selection_split':'validation','selected':selected,
         'train_class_counts':dict(Counter(train_y)),'candidates':history,
         'vocabulary_fit_split':'train','versions':{'scikit-learn':importlib.metadata.version('scikit-learn')}})
    for split,rr in [('train_diagnostic',rows),('validation_best',validation)]:
        pred=winner.predict([r['text'] for r in rr]).tolist()
        gold=[key(r['target'][column['code']]) for r in rr]
        report=classification_report(gold,pred,classes)
        report['predicted_class_counts']=dict(Counter(pred))
        report['constant_prediction']=len(set(pred))==1 and len(set(gold))>1
        dump(dest/f'{split}_metrics.json',{'split':'train' if split=='train_diagnostic' else 'validation',
             'diagnostic_only':split=='train_diagnostic','metrics':report})
        jsonl(dest/f'{split}_predictions.jsonl',[{'id':r['id'],'gold':r['target'][column['code']],
              'prediction':json.loads(v)} for r,v in zip(rr,pred)])
    return {'backend':'tfidf','selected':selected,'validation':classification_report(val_y,winner.predict(val_x).tolist(),classes)}


def train_hybrid(args):
    data=Path(args.data);schema=load(data/'schema.json')
    parts={s:read_jsonl(data/f'{s}.jsonl') for s in ['train','validation']}
    audit_training_parts(parts)
    out=new_output(args.output);epochs=int(args.epochs)
    if epochs<1 or epochs!=args.epochs:
        raise ValueError('spaCy epochs must be a positive integer')
    started=time.perf_counter();results={}
    for column in schema:
        code=column['code'];results[code]={}
        if args.task in {'ner','both'}:
            dest=out/code/'ner';dest.mkdir(parents=True)
            results[code]['ner']=train_spacy(parts['train'],parts['validation'],column,dest,epochs,args.seed)
        if args.task in {'classifier','both'}:
            dest=out/code/'classifier';dest.mkdir(parents=True)
            results[code]['classifier']=train_classifier(parts['train'],parts['validation'],column,dest,args.seed)
    dump(out/'schema.json',schema)
    dump(out/'run_config.json',{**vars(args),'backend':'hybrid','source_sha256':load(data/'manifest.json')['source_sha256']})
    dump(out/'validation_summary.json',{'selection_split':'validation','test_evaluated':False,'results':results,
         'seconds':time.perf_counter()-started,'versions':{**versions(),'spacy':importlib.metadata.version('spacy')}})


def evaluate_hybrid(args):
    import joblib
    data=Path(args.data);schema=load(data/'schema.json');rows=read_jsonl(data/f'{args.split}.jsonl')
    if load(Path(args.models)/'schema.json')!=schema:
        raise ValueError('Trained schema differs from evaluation schema')
    out=new_output(args.output);items={r['id']:{'id':r['id'],'prediction':{},'spans':[],'errors':[]} for r in rows}
    tasks=['ner','classifier'] if args.task=='both' else [args.task];started=time.perf_counter()
    ner_reports={}
    for column in schema:
        code=column['code']
        if 'ner' in tasks:
            import spacy
            nlp=spacy.load(Path(args.models)/code/'ner/model')
            pp=spacy_predictions(nlp,rows,code);ner_reports[code]=ner_report(rows,pp,code)
            for pred in pp:items[pred['id']]['spans'].extend(pred['spans'])
        if 'classifier' in tasks:
            model=joblib.load(Path(args.models)/code/'classifier/model.joblib')
            for row,pred in zip(rows,model.predict([r['text'] for r in rows])):
                items[row['id']]['prediction'][code]=json.loads(pred)
    metrics=value_metrics(rows,list(items.values()),schema) if 'classifier' in tasks else {'n':len(rows)}
    if 'ner' in tasks:
        gold=[(r['id'],a['label'],a['start'],a['end']) for r in rows for a in r['spans']]
        pred=[(r['id'],a['label'],a['start'],a['end']) for r in items.values() for a in r['spans']]
        metrics['ner_exact_spans']=span_metrics(gold,pred);metrics['ner_by_field']=ner_reports
    metrics['record_errors']=0
    save_metrics(out,metrics,{**vars(args),'backend':'hybrid','versions':versions(),
        'manifest':load(data/'manifest.json'),'diagnostic_only':args.split=='train','seconds':time.perf_counter()-started,
        'hybrid_versions':{name:importlib.metadata.version(name) for name in ['spacy','scikit-learn','joblib']}})
    jsonl(out/'predictions.jsonl',list(items.values()));dump(out/'errors.json',[])
    print(json.dumps({k:v for k,v in metrics.items() if k not in {'fields','ner_by_field'}},indent=2))

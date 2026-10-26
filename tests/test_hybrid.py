import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from common.data import dump,jsonl,load,read_jsonl
from common.hybrid import ner_selection,ner_report,classifier_pipeline,train_classifier,make_examples,train_hybrid,evaluate_hybrid
from common.encoder_training import class_balance,weighted_trainer_class

COL={'code':'AG','name':'Гипертония','data_type':'bool','values':[False,True,None]}


def sample_rows(prefix='train'):
    rows=[]
    for i,(text,value) in enumerate([('Гипертония есть.',True),('Гипертонию отрицает.',False),('Сведений нет.',None)]*2):
        spans=[] if value is None else [{'label':'AG','start':0,'end':10,'text':text[:10],'result':value}]
        rows.append({'id':prefix+str(i),'document_id':prefix+'doc'+str(i),'text':text,'target':{'AG':value},'spans':spans})
    return rows


def test_ner_report_keeps_unaligned_gold_and_false_positives():
    rows=[{'id':'h','text':'АГ нет','spans':[{'label':'AG','start':1,'end':2}]}]
    pp=[{'id':'h','spans':[{'label':'AG','start':0,'end':2}]}]
    report=ner_report(rows,pp,'AG')
    assert report['ner_f1']==0
    assert report['ner_exact_spans']['fp']==report['ner_exact_spans']['fn']==1
    assert report['ner_entity_characters']['f1']>0


def test_lexicographic_ner_selection_preserves_primary_metric():
    def report(f1,char,fp):return {'ner_f1':f1,'ner_entity_characters':{'f1':char},'ner_exact_spans':{'fp':fp}}
    assert ner_selection(report(.1,0,1000))>ner_selection(report(.09,1,0))
    assert ner_selection(report(0,.3,400))>ner_selection(report(0,.2,700))


def test_softened_ner_weight_ratio():
    rows=[{'labels':[0]*100+[1]+[2]*10}]
    _,balanced=class_balance(rows,['O','B','I'],'ner')
    report,soft=class_balance(rows,['O','B','I'],'ner',method='sqrt')
    assert report['method']=='sqrt_inverse_frequency'
    assert soft[1]/soft[0]==pytest.approx(10)
    assert balanced[1]/balanced[0]==pytest.approx(100)


def test_classifier_vocabulary_fitted_only_on_train(tmp_path):
    pytest.importorskip('sklearn')
    rows=sample_rows();validation=sample_rows('val')
    validation[0]['text']+=' ValidationSecretUnseen'
    train_classifier(rows,validation,COL,tmp_path,42)
    import joblib
    pipeline=joblib.load(tmp_path/'model.joblib')
    vocabulary=pipeline.named_steps['tfidf'].transformer_list[0][1].vocabulary_
    assert 'validationsecretunseen' not in vocabulary
    assert 'отрицает' in vocabulary  # Negation is retained, not removed as a stopword.
    assert load(tmp_path/'training_metrics.json')['vocabulary_fit_split']=='train'
    assert load(tmp_path/'train_diagnostic_metrics.json')['metrics']['accuracy']==1
    assert len(read_jsonl(tmp_path/'validation_best_predictions.jsonl'))==6


def test_spacy_alignment_changes_only_training_copy():
    spacy=pytest.importorskip('spacy')
    rows=[{'id':'h','text':'Гипертония нет','spans':[{'label':'AG','start':1,'end':10}]}]
    original=json.dumps(rows)
    examples,report=make_examples(spacy.blank('ru'),rows,'AG')
    assert report and json.dumps(rows)==original
    assert examples[0].reference.ents[0].start_char==0


def test_hybrid_train_never_reads_test_file(tmp_path):
    pytest.importorskip('spacy')
    data=tmp_path/'data';data.mkdir()
    train=sample_rows();val=sample_rows('val')
    # Distinct wording for source leakage audit, identical classes.
    for i,r in enumerate(train):r['text']+=' train '+str(i)
    for i,r in enumerate(val):r['text']+=' valid '+str(i)
    jsonl(data/'train.jsonl',train);jsonl(data/'validation.jsonl',val)
    (data/'test.jsonl').write_text('NOT JSON')
    dump(data/'schema.json',[COL]);dump(data/'manifest.json',{'source_sha256':'synthetic','splits':{'train':[r['id'] for r in train],'validation':[r['id'] for r in val]}})
    out=tmp_path/'models'
    train_hybrid(SimpleNamespace(data=str(data),output=str(out),epochs=2,seed=42,task='both',backend='hybrid'))
    assert (data/'test.jsonl').read_text()=='NOT JSON'
    assert load(out/'validation_summary.json')['test_evaluated'] is False
    evaluate_hybrid(SimpleNamespace(data=str(data),models=str(out),output=str(tmp_path/'evaluation'),split='validation',task='both'))
    assert load(tmp_path/'evaluation/metrics.json')['n']==6
    for task in ['ner','classifier']:
        assert (out/'AG'/task/'train_diagnostic_metrics.json').exists()


def test_transformer_tie_break_prefers_progress_not_first_epoch(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import TrainingArguments
    model=torch.nn.Linear(2,3)
    args=TrainingArguments(output_dir=str(tmp_path),report_to=[],metric_for_best_model='eval_ner_f1',greater_is_better=True)
    trainer=weighted_trainer_class()(model=model,args=args,class_weights=[1,1,1])
    trainer.state.global_step=12
    assert trainer._determine_best_metric({'eval_ner_f1':0,'eval_ner_entity_token_f1':.18,'eval_loss':1.1},None)
    trainer.state.global_step=60
    assert trainer._determine_best_metric({'eval_ner_f1':0,'eval_ner_entity_token_f1':.29,'eval_loss':.85},None)
    assert trainer.state.best_global_step==60 and trainer.state.best_metric==0
    assert not trainer._determine_best_metric({'eval_ner_f1':0,'eval_ner_entity_token_f1':.2,'eval_loss':.1},None)

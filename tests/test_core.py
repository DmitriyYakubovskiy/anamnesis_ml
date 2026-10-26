import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import validate,split_groups,key
from common.metrics import parse_answer,value_metrics,span_metrics
from common.encoder import bio_labels,decode_bio
from common.generative import sft_example


def records():
    return [{'id':str(i),'document_id':'doc'+str(i//2),'text':f'Пациент {i}: АГ нет 🚬.', 'completed':1,
             'columns':[{'code':'AG','name':'Гипертония','data_type':'bool','active':True,'value':v,'state':'no_data' if v is None else 'value'}],
             'annotations':[]} for i,v in enumerate([True,False,None]*4)]


def test_null_false_and_zero_are_distinct():
    assert len({key(False),key(0),key(None)})==3


def test_export_states_and_inactive_columns():
    rows=records();rows[0]['columns'].append({'active':False,'code':'UNUSED','value':None})
    items,schema,excluded=validate(rows)
    assert not excluded and len(schema)==1
    assert items[1]['target']['AG'] is False and items[2]['target']['AG'] is None
    assert all('UNUSED' not in x['target'] for x in items)


def test_legacy_null_is_excluded():
    rows=records();rows[2]['columns'][0].pop('state')
    items,_,excluded=validate(rows)
    assert len(items)==11 and excluded==[{'id':'2','reason':'unreviewed_active_column'}]


@pytest.mark.parametrize('field,value',[('state','no_data'),('state','wrong'),('value','true'),('value',1)])
def test_invalid_column_rejected(field,value):
    rows=records();rows[0]['columns'][0][field]=value
    with pytest.raises(ValueError):validate(rows)


def test_unicode_offsets():
    rows=records();r=rows[0];start=r['text'].index('АГ')
    r['annotations']=[{'label':'AG','start':start,'end':start+2,'text':'АГ','result':True,'result_state':'value'}]
    assert validate(rows)[0][0]['spans'][0]['text']=='АГ'
    r['annotations'][0]['end']+=1
    with pytest.raises(ValueError):validate(rows)


def test_overlap_rejected():
    rows=records();r=rows[0]
    r['annotations']=[{'label':'AG','start':0,'end':7,'text':r['text'][:7],'result':True}, {'label':'AG','start':1,'end':3,'text':r['text'][1:3],'result':False}]
    with pytest.raises(ValueError,match='Overlapping'):validate(rows)


def test_unreviewed_annotation_is_excluded():
    rows=records();r=rows[0];r['annotations']=[{'label':'AG','start':0,'end':7,'text':r['text'][:7],'result':None}]
    assert validate(rows)[2][0]['reason']=='unreviewed_span_result'


def test_grouping_reproducible_and_duplicate_safe():
    rows=records();rows[-1]['text']=rows[0]['text']
    items,_,_=validate(rows)
    a,b=split_groups(items),split_groups(list(reversed(items)))
    assert {s:[r['id'] for r in rr] for s,rr in a.items()}=={s:[r['id'] for r in rr] for s,rr in b.items()}
    assert sorted(r['id'] for rr in a.values() for r in rr)==sorted(r['id'] for r in items)
    for left,right in [('train','validation'),('train','test'),('test','validation')]:
        assert not {r['document_id'] for r in a[left]} & {r['document_id'] for r in a[right]}
        assert not {r['text'] for r in a[left]} & {r['text'] for r in a[right]}


@pytest.mark.parametrize('answer',['{}','{"AG":0}','{"AG":"true"}','{"AG":true,"extra":1}','```json\n{"AG":true}\n```'])
def test_malformed_output_not_repaired(answer):
    _,schema,_=validate(records())
    with pytest.raises((ValueError,json.JSONDecodeError)):parse_answer(answer,schema)


def test_null_prediction_is_valid_but_missing_is_not():
    rows,schema,_=validate(records())
    r=rows[2]
    assert parse_answer('{"AG":null}',schema)==r['target']
    metrics=value_metrics([r],[{'id':r['id'],'prediction':None}],schema)
    assert metrics['exact_match']==0 and metrics['invalid_rate']==1
    metrics=value_metrics([r],[{'id':r['id'],'prediction':{'AG':None}}],schema)
    assert metrics['exact_match']==1 and metrics['invalid_rate']==0


def test_span_f1_penalizes_boundary_error():
    m=span_metrics([('h','AG',0,2)],[('h','AG',0,3)])
    assert m['f1']==0 and m['fp']==m['fn']==1


def test_bio_roundtrip_and_expansion():
    offsets=[(0,0),(0,2),(3,5),(6,9),(0,0)]
    ids,expanded=bio_labels(offsets,[{'start':0,'end':5}])
    assert ids==[-100,1,2,0,-100] and not expanded
    assert decode_bio(ids,offsets)==[(0,5)]
    assert bio_labels(offsets,[{'start':1,'end':5}])[1]


def test_bio_same_token_collision_is_error():
    with pytest.raises(ValueError):bio_labels([(0,4)],[{'start':0,'end':1},{'start':2,'end':4}])


def test_sft_masks_prompt_and_protects_answer():
    class Tokenizer:
        eos_token_id=99
        def apply_chat_template(self,*args,**kwargs):return [1,2,3]
        def encode(self,*args,**kwargs):return [4,5]
    rows,schema,_=validate(records())
    r=sft_example(Tokenizer(),rows[0],schema,6)
    assert r['labels']==[-100,-100,-100,4,5,99]
    with pytest.raises(ValueError):sft_example(Tokenizer(),rows[0],schema,5)


def test_local_lmstudio_evaluation_keeps_failures(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler,HTTPServer
    from threading import Thread
    from common.data import dump,prepare,load,read_jsonl
    from common.generative import evaluate
    source=tmp_path/'source.json';dump(source,records());data=tmp_path/'data';prepare(source,data)
    test=read_jsonl(data/'test.jsonl')
    targets={r['text']:r['target'] for r in test}
    class Handler(BaseHTTPRequestHandler):
        count=0
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/v1/chat/completions'
            text=body['messages'][-1]['content'];Handler.count+=1
            answer='invalid json' if Handler.count==1 else json.dumps(targets[text])
            out=json.dumps({'choices':[{'message':{'content':answer}}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(out)));self.end_headers();self.wfile.write(out)
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        out=tmp_path/'eval'
        monkeypatch.setattr(sys,'argv',['evaluate','--data',str(data),'--base-url',f'http://127.0.0.1:{server.server_port}/v1','--model','mock','--output',str(out)])
        evaluate('qwen')
        report=load(out/'metrics.json')
        assert report['n']==len(test) and report['invalid_count']==1
        assert len(read_jsonl(out/'predictions.jsonl'))==len(test)
        assert report['exact_match']==(len(test)-1)/len(test)
        assert (out/'field_metrics.csv').exists()
    finally:
        server.shutdown();server.server_close();thread.join()


def test_unsafe_column_code_rejected():
    rows=records();rows[0]['columns'][0]['code']='../../models'
    with pytest.raises(ValueError,match='column code'):validate(rows)


def test_sft_padding_is_ignored_not_eos():
    torch=pytest.importorskip('torch')
    from common.generative import SFTCollator
    class Tok:pad_token_id=99
    batch=SFTCollator(Tok())([
        {'input_ids':[1,2,99],'attention_mask':[1,1,1],'labels':[-100,2,99]},
        {'input_ids':[1,99],'attention_mask':[1,1],'labels':[-100,99]},
    ])
    assert batch['labels'].tolist()==[[-100,2,99],[-100,99,-100]]
    assert batch['attention_mask'].tolist()==[[1,1,1],[1,1,0]]

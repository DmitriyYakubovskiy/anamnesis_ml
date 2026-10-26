"""F1 selection, train-only weighting, ignored labels and diagnostic reports."""
import copy
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from common.encoder_training import class_balance,metric_function,summarize_predictions,weighted_trainer_class,audit_training_parts

COLUMN={'code':'AG','values':[False,True,None]}


def test_classifier_weights_use_train_frequencies():
    train=[{'labels':i} for i in [0,1,1,1,2,2]]
    report,weights=class_balance(train,['false','true','null'],'classifier')
    assert weights==pytest.approx([2,2/3,1])
    assert report['counts']=={'false':1,'true':3,'null':2}
    assert report['source_split']=='train'


def test_ner_weights_ignore_special_padding_labels():
    train=[{'labels':[-100,0,0,0,1,2,-100]},{'labels':[0,0,2,-100]}]
    report,weights=class_balance(train,['O','B','I'],'ner')
    assert report['counts']=={'O':5,'B':1,'I':2}
    assert report['total_targets']==8
    assert weights==pytest.approx([8/15,8/3,8/6])
    assert weights[1]>weights[0]


def test_missing_classes_finite_and_reported():
    report,weights=class_balance([{'labels':[0,-100]}],['O','B','I'],'ner')
    assert weights==[1,0,0] and report['missing_classes']==['B','I']
    assert class_balance([{'labels':0}],['a','b'],'classifier',False)[1]==[1,1]
    with pytest.raises(ValueError):class_balance([{'labels':[-100]}],['O'],'ner')


def test_classifier_f1_exposes_majority_collapse():
    rows=[{'id':str(i),'target':{'AG':v}} for i,v in enumerate([True]*5+[False]+[None]*3)]
    flat,details,pred=summarize_predictions([1]*9,[1]*5+[0]+[2]*3,rows,[],COLUMN,'classifier')
    assert flat['accuracy']==pytest.approx(5/9)
    assert flat['macro_f1']==pytest.approx(5/21)
    assert flat['class_0_recall']==flat['class_2_recall']==0
    assert len(pred)==9
    assert details['constant_prediction'] is True
    assert details['predicted_class_counts']=={'true':9}


def test_ner_exact_f1_uses_original_not_expanded_boundaries():
    row={'id':'h','text':'АГда','spans':[{'label':'AG','start':1,'end':4,'text':'Гда'}]}
    offsets=[[(0,0),(0,2),(2,4),(0,0)]]
    flat,details,pred=summarize_predictions([[0,1,2,0]], [[-100,1,2,-100]], [row],offsets,COLUMN,'ner')
    assert flat['ner_f1']==0
    assert flat['ner_token_aligned_f1']==1
    assert flat['ner_entity_token_f1']==1
    assert details['exact_spans']['fp']==details['exact_spans']['fn']==1


def test_ner_all_o_has_zero_entity_f1():
    row={'id':'h','text':'АГ нет','spans':[{'label':'AG','start':0,'end':2,'text':'АГ'}]}
    off=[[(0,0),(0,2),(3,6),(0,0)]]
    flat,_,_=summarize_predictions([[0,0,0,0]], [[-100,1,0,-100]], [row],off,COLUMN,'ner')
    assert flat['ner_f1']==0 and flat['ner_fn']==1 and flat['token_accuracy']==.5


def test_metric_callback_handles_padding_and_argmax_ids():
    np=pytest.importorskip('numpy')
    rows=[{'id':'h','text':'АГ нет','spans':[{'label':'AG','start':0,'end':2,'text':'АГ'}]}]
    off=[[(0,0),(0,2),(3,6),(0,0)]]
    callback=metric_function(rows,off,COLUMN,'ner')
    ev=SimpleNamespace(predictions=np.array([[0,1,0,0,2,2]]),label_ids=np.array([[-100,1,0,-100,-100,-100]]))
    assert callback(ev)['ner_f1']==1
    logits=np.eye(3)[ev.predictions]
    assert callback(SimpleNamespace(predictions=logits,label_ids=ev.label_ids))['ner_f1']==1


def test_weighted_loss_matches_pytorch_and_backprop(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import TrainingArguments
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.tensor([[[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]]]))
        def forward(self,input_ids=None):return SimpleNamespace(logits=self.logits)
    model=Model();weights=[.5,3.,2.]
    trainer=weighted_trainer_class()(model=model,args=TrainingArguments(output_dir=str(tmp_path),report_to=[]),class_weights=weights)
    labels=torch.tensor([[0,2,-100]])
    loss=trainer.compute_loss(model,{'input_ids':torch.ones(1,3,dtype=torch.long),'labels':labels})
    expected=torch.nn.functional.cross_entropy(model.logits.reshape(-1,3),labels.reshape(-1),weight=torch.tensor(weights),ignore_index=-100)
    assert torch.allclose(loss,expected)
    loss.backward();assert model.logits.grad.abs().sum()>0
    assert trainer.model_accepts_loss_kwargs is False
    ignored=trainer.compute_loss(model,{'labels':torch.full((1,3),-100)})
    assert torch.isfinite(ignored) and ignored.item()==0


@pytest.mark.parametrize('field',['id','document_id','text'])
def test_train_validation_leakage_rejected(field):
    parts={'train':[{'id':'a','document_id':'d1','text':'текст один'}],
           'validation':[{'id':'b','document_id':'d2','text':'текст два'}]}
    audit_training_parts(parts)
    parts['validation'][0][field]=parts['train'][0][field]
    with pytest.raises(ValueError,match='leakage'):audit_training_parts(parts)


def test_best_checkpoint_is_chosen_by_f1_and_restored(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import BertConfig,BertForSequenceClassification,TrainingArguments
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    model=BertForSequenceClassification(BertConfig(vocab_size=10,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,num_labels=3))
    examples=[{'input_ids':[1,2,3],'attention_mask':[1,1,1],'labels':0},
              {'input_ids':[3,2,1],'attention_mask':[1,1,1],'labels':1}]
    epochs=[]
    def metrics(prediction):
        epochs.append(1)
        return {'macro_f1':.9 if len(epochs)==1 else .1}
    args=TrainingArguments(output_dir=str(tmp_path),num_train_epochs=2,per_device_train_batch_size=2,
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_macro_f1',
        greater_is_better=True,report_to=[],disable_tqdm=True,save_total_limit=2,use_cpu=True)
    trainer=weighted_trainer_class()(model=model,args=args,class_weights=[1,1,1],train_dataset=examples,eval_dataset=examples,compute_metrics=metrics)
    trainer.train()
    assert trainer.state.best_metric==.9
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-1')
    weights=load_file(str(Path(trainer.state.best_model_checkpoint)/'model.safetensors'))
    assert torch.equal(trainer.model.classifier.weight.detach().cpu(),weights['classifier.weight'])

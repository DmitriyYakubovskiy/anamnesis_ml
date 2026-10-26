"""CPU end-to-end smoke test using RANDOM tiny models; not meaningful quality metrics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.data import dump,prepare


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    import torch
    torch.set_num_threads(1)
    from transformers import BertConfig,BertForMaskedLM,BertTokenizerFast,Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);bert=temp/'bert';bert.mkdir()
        (bert/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nпациент\nаг\nнет\nда\nсведения\nотсутствуют\n.\n',encoding='utf-8')
        tok=BertTokenizerFast(vocab_file=str(bert/'vocab.txt'),do_lower_case=True)
        tok.save_pretrained(bert)
        BertForMaskedLM(BertConfig(vocab_size=len(tok),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=2048)).save_pretrained(bert)
        llm=temp/'llm';llm.mkdir()
        raw=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'user':3,'assistant':4,'system':5},unk_token='[UNK]'))
        raw.pre_tokenizer=pre_tokenizers.Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        tok.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        tok.save_pretrained(llm)
        Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=4096,eos_token_id=2,pad_token_id=1)).save_pretrained(llm)
        rows=[]
        for i,v in enumerate([True,False,None]*3):
            text='пациент аг да.' if v is True else 'пациент аг нет.' if v is False else 'сведения отсутствуют.'
            text += ' '+str(i)
            spans=[] if v is None else [{'label':'AG','start':8,'end':10,'text':'аг','result':v,'result_state':'value'}]
            rows.append({'id':str(i),'document_id':'doc'+str(i),'text':text,'completed':1,'columns':[{'code':'AG','name':'Гипертония','data_type':'bool','active':True,'value':v,'state':'no_data' if v is None else 'value'}],'annotations':spans})
        source=temp/'input.json';dump(source,rows);data=temp/'data';prepare(source,data)
        common=['--data',data,'--epochs','1','--max-steps','1','--batch','1']
        # A corrupt test file must not affect fitting/selection/diagnostics.
        test_file=data/'test.jsonl';test_bytes=test_file.read_bytes();test_file.write_text('THIS IS NOT JSON')
        run('ner_classifier/train.py',*common,'--model',bert,'--output',temp/'enc')
        assert test_file.read_text()=='THIS IS NOT JSON'
        test_file.write_bytes(test_bytes)
        for task,metric in [('ner','eval_ner_f1'),('classifier','eval_macro_f1')]:
            dest=temp/'enc'/'AG'/task
            report=json.loads((dest/'training_metrics.json').read_text())
            assert report['config']['selection_metric']==metric
            assert any(metric in row for row in report['train'])
            for split in ['train_diagnostic','validation_best']:
                assert (dest/(split+'_metrics.json')).exists()
                assert (dest/(split+'_predictions.jsonl')).exists()
            assert json.loads((dest/'class_weights.json').read_text())['source_split']=='train'
        run('ner_classifier/evaluate.py','--data',data,'--models',temp/'enc','--output',temp/'enc_train','--split','train')
        assert json.loads((temp/'enc_train/metrics.json').read_text())['metadata']['diagnostic_only'] is True
        run('ner_classifier/evaluate.py','--data',data,'--models',temp/'enc','--output',temp/'enc_eval','--split','test')
        run('Lora/train.py',*common,'--model',llm,'--output',temp/'lora','--accumulation','1','--rank','2')
        run('Lora/evaluate.py','--data',data,'--adapter',temp/'lora/model','--output',temp/'lora_eval','--max-new-tokens','2')
        run('Lora/merge.py','--adapter',temp/'lora/model','--output',temp/'merged')
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        base=AutoModelForCausalLM.from_pretrained(llm)
        adapted=PeftModel.from_pretrained(base,temp/'lora/model').eval()
        merged=AutoModelForCausalLM.from_pretrained(temp/'merged').eval()
        x=torch.tensor([[3,4,5]])
        with torch.no_grad():assert torch.allclose(adapted(x).logits,merged(x).logits,atol=1e-5)
        run('qwen/train.py',*common,'--model',llm,'--output',temp/'full','--accumulation','1','--confirm-full-finetune')
        run('qwen/evaluate.py','--data',data,'--model',llm,'--output',temp/'baseline','--max-new-tokens','2')
        run('compare.py',temp/'enc_eval/metrics.json',temp/'lora_eval/metrics.json',temp/'baseline/metrics.json','--output',temp/'comparison.json')
        for f in ['enc_eval','lora_eval','baseline']:
            assert (temp/f/'metrics.json').exists() and (temp/f/'predictions.jsonl').exists()
        print('SMOKE PASS: NER + classifier, LoRA train/eval, merge equivalence, full SFT, plain Qwen eval. Random tiny weights, no quality claim.')


if __name__=='__main__':main()

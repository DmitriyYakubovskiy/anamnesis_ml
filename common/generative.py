"""Answer-only SFT and shared Qwen/adapter evaluation; model imports are lazy."""
import argparse
import json
from pathlib import Path
import time
from common.data import ROOT, load, read_jsonl, jsonl, dump, key
from common.prompts import messages
from common.metrics import parse_answer, value_metrics, save_metrics
from common.runtime import DEFAULT_LLM, precision, new_output, training_args, save_training, versions, timed_run, timed_phase


def sft_example(tokenizer, row, schema, max_length):
    prompt = tokenizer.apply_chat_template(messages(row['text'], schema), tokenize=True, add_generation_prompt=True)
    answer = tokenizer.encode(key(row['target']), add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt)+len(answer) > max_length:
        raise ValueError(f"Record {row['id']} exceeds max_length; increase it, do not truncate gold answers")
    return {'input_ids': prompt+answer, 'attention_mask': [1]*(len(prompt)+len(answer)),
            'labels': [-100]*len(prompt)+answer}


class SFTCollator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id
    def __call__(self, features):
        import torch
        n = max(len(x['input_ids']) for x in features)
        return {key: torch.tensor([x[key]+[self.pad if key == 'input_ids' else -100 if key == 'labels' else 0]*(n-len(x[key])) for x in features]) for key in ['input_ids','attention_mask','labels']}


@timed_run('train')
def train(mode):
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(ROOT/('Lora' if mode == 'lora' else 'qwen')/'data'))
    p.add_argument('--model', default=DEFAULT_LLM)
    p.add_argument('--revision', default='main', help='Prefer an immutable HF commit for repeatability')
    p.add_argument('--output', default=str(ROOT/'runs'/('lora' if mode == 'lora' else 'qwen_full')))
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--lr', type=float, default=2e-4 if mode == 'lora' else 1e-5)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--four-bit', action='store_true')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--confirm-full-finetune', action='store_true')
    a = p.parse_args()
    if mode == 'full' and not a.confirm_full_finetune:
        p.error('Plain Qwen requires no training. For optional full SFT use --confirm-full-finetune (high memory usage).')
    if mode == 'full' and a.four_bit:
        p.error('Four-bit training is supported only for LoRA')
    out = new_output(a.output)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, set_seed, BitsAndBytesConfig
    set_seed(a.seed)
    data = Path(a.data)
    schema = load(data/'schema.json')
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    datasets = {s: [sft_example(tokenizer, r, schema, a.max_length) for r in read_jsonl(data/f'{s}.jsonl')] for s in ['train','validation']}
    kwargs = {'torch_dtype': precision(), 'revision': a.revision}
    if a.four_bit:
        if not torch.cuda.is_available():
            p.error('--four-bit requires NVIDIA CUDA and requirements-4bit.txt')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=precision()), device_map={'': torch.cuda.current_device()})
    model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
    base_commit = getattr(model.config, '_commit_hash', None)
    model.config.use_cache = False
    if mode == 'lora':
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if a.four_bit:
            model = prepare_model_for_kbit_training(model)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=a.rank, lora_alpha=2*a.rank, lora_dropout=.05, target_modules='all-linear', bias='none'))
        model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable()
    trainer = Trainer(model=model, args=training_args(out,a.epochs,a.lr,a.batch,a.accumulation,a.seed,a.max_steps),
                      train_dataset=datasets['train'], eval_dataset=datasets['validation'], data_collator=SFTCollator(tokenizer), processing_class=tokenizer)
    with timed_phase('fit_and_validation', backend=mode):
        result = trainer.train()
    config = {**vars(a), 'mode': mode, 'base_commit': base_commit, 'source_sha256': load(data/'manifest.json')['source_sha256']}
    # PEFT config's revision is used later to load exactly the same base.
    if mode == 'lora':
        model.peft_config['default'].revision = base_commit or a.revision
    model.config.use_cache = True
    save_training(trainer,tokenizer,out,config)
    dump(out/'run_config.json',config)
    dump(out/'train_summary.json',result.metrics)
    dump(out/'model'/'schema.json',schema)


def local_generator(model_name, revision='main', adapter=None, max_length=4096, max_new_tokens=256, four_bit=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = {'torch_dtype': precision(), 'revision': revision, 'device_map': 'auto'}
    if adapter:
        from peft import PeftConfig
        cfg = PeftConfig.from_pretrained(adapter)
        # Adapter determines the base: never attach DeepSeek weights to Qwen.
        model_name, revision = cfg.base_model_name_or_path, cfg.revision or revision
        kwargs['revision'] = revision
    if four_bit:
        if not torch.cuda.is_available():
            raise ValueError('Four-bit inference requires CUDA')
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=precision())
    tok = AutoTokenizer.from_pretrained(adapter or model_name, revision=None if adapter else revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    def generate(text, schema):
        inp = tok.apply_chat_template(messages(text,schema), tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
        if inp['input_ids'].shape[-1]+max_new_tokens > max_length:
            raise ValueError('Input plus generation budget exceeds --max-length; input was not truncated')
        with torch.inference_mode():
            output = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(output[0,inp['input_ids'].shape[-1]:],skip_special_tokens=True)
    generate.model_info = {'base_model': model_name, 'revision': revision, 'resolved_commit': getattr(model.config, '_commit_hash', None), 'adapter': adapter}
    return generate


def api_generator(base_url, model, timeout=180):
    import os
    from urllib.request import Request, urlopen
    def generate(text,schema):
        body=json.dumps({'model':model,'messages':messages(text,schema),'temperature':0,'max_tokens':512,'stream':False}).encode()
        req=Request(base_url.rstrip('/')+'/chat/completions',data=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+os.getenv('LM_API_KEY','lm-studio')})
        with urlopen(req,timeout=timeout) as response:
            return json.load(response)['choices'][0]['message']['content']
    generate.model_info = {'base_url': base_url, 'served_model': model}
    return generate


@timed_run('evaluate')
def evaluate(folder):
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(ROOT/folder/'data'))
    p.add_argument('--model',default=DEFAULT_LLM)
    p.add_argument('--revision',default='main')
    p.add_argument('--adapter')
    p.add_argument('--base-url',help='Explicit OpenAI-compatible local LM Studio endpoint, e.g. http://localhost:1234/v1')
    p.add_argument('--output',default=str(ROOT/'runs'/('lora_eval' if folder=='Lora' else 'qwen_baseline')))
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--max-length',type=int,default=4096)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--four-bit',action='store_true')
    a=p.parse_args()
    if a.base_url and a.adapter:
        p.error('--adapter only applies to local Transformers; load the merged model into LM Studio instead')
    data=Path(a.data);schema=load(data/'schema.json');rows=read_jsonl(data/f'{a.split}.jsonl')
    for candidate in [a.adapter, a.model if not a.base_url else None]:
        if candidate and (Path(candidate)/'schema.json').exists() and load(Path(candidate)/'schema.json') != schema:
            raise ValueError('Model schema differs from evaluation data')
    out=new_output(a.output)
    with timed_phase('model_load', backend='api' if a.base_url else 'transformers'):
        generator=api_generator(a.base_url,a.model) if a.base_url else local_generator(a.model,a.revision,a.adapter,a.max_length,a.max_new_tokens,a.four_bit)
    predictions=[];started=time.perf_counter()
    with timed_phase('inference', backend='api' if a.base_url else 'transformers', samples=len(rows)):
        for r in rows:
            t=time.perf_counter();item={'id':r['id'],'prediction':None,'raw':'','error':None}
            try:
                item['raw']=generator(r['text'],schema)
                item['prediction']=parse_answer(item['raw'],schema)
            except Exception as e:
                item['error']=f'{type(e).__name__}: {e}'
            item['seconds']=time.perf_counter()-t
            predictions.append(item)
            jsonl(out/'predictions.jsonl',predictions)
    metrics=value_metrics(rows,predictions,schema)
    save_metrics(out,metrics,{**vars(a),'versions':versions(),'seconds':time.perf_counter()-started,'model_info':generator.model_info,'manifest':load(data/'manifest.json')})
    dump(out/'errors.json',[r for r in predictions if r['error']])
    print(json.dumps({k:v for k,v in metrics.items() if k!='fields'},indent=2))

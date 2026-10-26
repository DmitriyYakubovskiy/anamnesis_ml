"""Regression for PEFT's second dispatch with disk offload; no model downloads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest


def test_lora_second_dispatch_can_offload_to_disk(tmp_path, monkeypatch):
    torch=pytest.importorskip('torch')
    pytest.importorskip('peft')
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers
    from peft import LoraConfig, PeftModel, get_peft_model
    import peft.peft_model as peft_loading
    from common.generative import local_generator

    base=tmp_path/'base';adapter=tmp_path/'adapter'
    torch.manual_seed(42)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=4096,eos_token_id=2,pad_token_id=1))
    model.save_pretrained(base)
    model=Qwen3ForCausalLM.from_pretrained(base)
    raw=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'user':3,'assistant':4,'system':5,'да':6,'нет':7},unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]',
                               model_input_names=['input_ids','attention_mask'])
    tok.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
    for name,weight in model.named_parameters():
        if 'lora_B' in name:
            weight.data.fill_(.05)
    model.save_pretrained(adapter);tok.save_pretrained(adapter)

    loaded=[];adapter_calls=[];base_calls=[]
    original_adapter=PeftModel.from_pretrained
    original_base=AutoModelForCausalLM.from_pretrained
    def capture_adapter(*args,**kwargs):
        adapter_calls.append(kwargs.copy())
        result=original_adapter(*args,**kwargs);loaded.append(result);return result
    def capture_base(*args,**kwargs):
        base_calls.append(kwargs.copy())
        return original_base(*args,**kwargs)
    monkeypatch.setattr(PeftModel,'from_pretrained',capture_adapter)
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',capture_base)
    local_generator(str(base),adapter=str(adapter),device_map='cpu',offload_folder=tmp_path/'cpu')
    x=torch.tensor([[3,6,7]])
    with torch.inference_mode():
        expected=loaded[-1](x).logits.clone()

    # The base initially fits; PEFT's subsequent dispatch offloads one layer.
    placement={
        'base_model.model.model.embed_tokens':'cpu',
        'base_model.model.model.layers.0':'cpu',
        'base_model.model.model.layers.1':'disk',
        'base_model.model.model.norm':'cpu',
        'base_model.model.model.rotary_emb':'cpu',
        'base_model.model.lm_head':'cpu',
    }
    monkeypatch.setattr(peft_loading,'infer_auto_device_map',lambda *a,**kw: placement)
    with pytest.raises(ValueError,match='offload_dir'):
        local_generator(str(base),adapter=str(adapter))
    offload=tmp_path/'выгрузка'
    generate=local_generator(str(base),adapter=str(adapter),offload_folder=offload)
    assert base_calls[-1]['offload_folder']==adapter_calls[-1]['offload_folder']==str(offload.resolve())
    assert base_calls[-1]['dtype']==torch.float32
    assert (offload/'index.json').exists()
    assert 'disk' in generate.model_info['device_map'].values()
    with torch.inference_mode():
        actual=loaded[-1](x).logits
    assert torch.allclose(expected,actual,atol=1e-5)

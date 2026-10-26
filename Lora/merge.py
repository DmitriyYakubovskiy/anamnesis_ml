"""Merge an adapter into its ORIGINAL nonquantized base model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from common.runtime import new_output, precision
from common.data import dump


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    from peft import PeftConfig,PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    cfg=PeftConfig.from_pretrained(a.adapter)
    # Deliberately no --base override: architecture AND original weights must match.
    base=AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path,revision=cfg.revision,torch_dtype=precision(),device_map='auto')
    model=PeftModel.from_pretrained(base,a.adapter)
    merged=model.merge_and_unload(safe_merge=True)
    out=new_output(a.output)
    merged.save_pretrained(out,safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)
    dump(out/'merge_info.json',{'adapter':a.adapter,'base':cfg.base_model_name_or_path,'revision':cfg.revision})
    schema=Path(a.adapter)/'schema.json'
    if schema.exists():
        (out/'schema.json').write_bytes(schema.read_bytes())


if __name__=='__main__':
    main()

from pathlib import Path
import importlib.metadata
from common.data import dump

DEFAULT_LLM = 'Qwen/Qwen3-4B-Instruct-2507'
DEFAULT_ENCODER = 'cointegrated/rubert-tiny2'


def versions():
    result = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def precision():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output already has files; choose a new run directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_args(out, epochs, lr, batch, accumulation, seed, max_steps=-1):
    import torch
    from transformers import TrainingArguments
    return TrainingArguments(output_dir=str(out/'checkpoints'), num_train_epochs=epochs,
        learning_rate=lr, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accumulation, eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=2, logging_steps=1, report_to=[], seed=seed, data_seed=seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), fp16=False,
        dataloader_num_workers=0, max_steps=max_steps)


def save_training(trainer, tokenizer, out, config):
    trainer.save_model(str(out/'model'))
    tokenizer.save_pretrained(str(out/'model'))
    dump(out/'training_metrics.json', {'train': trainer.state.log_history, 'best_checkpoint': trainer.state.best_model_checkpoint,
                                      'config': config, 'versions': versions()})

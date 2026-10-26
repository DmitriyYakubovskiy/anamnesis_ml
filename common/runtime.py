from pathlib import Path
import importlib.metadata
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import time
from common.data import dump, load

DEFAULT_LLM = 'Qwen/Qwen3-4B-Instruct-2507'
DEFAULT_ENCODER = 'cointegrated/rubert-tiny2'
_active_timer = ContextVar('run_timer', default=None)


class RunTimer:
    """Wall time from entry into train/evaluate, including cold loading and export."""
    def __init__(self, operation):
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.operation = operation
        self.output = None
        self.phases = []

    def report(self, status):
        return {'operation': self.operation, 'status': status,
                'started_at': self.started_at,
                'finished_at': datetime.now(timezone.utc).isoformat() if status != 'running' else None,
                'total_seconds': time.perf_counter() - self.started,
                'scope': 'entrypoint_body_including_imports_data_model_loading_work_and_export',
                'phases': list(self.phases)}

    def finish(self, status):
        report = self.report(status)
        if self.output is not None:
            dump(self.output/'timing.json', report)
            # Preserve existing quality metrics and legacy metadata.seconds.
            filenames = ['metrics.json', 'training_metrics.json', 'validation_summary.json'] if status == 'completed' else []
            for filename in filenames:
                path = self.output/filename
                if path.exists():
                    data = load(path)
                    if filename == 'metrics.json':
                        data.setdefault('metadata', {})['timing'] = report
                    else:
                        data['timing'] = report
                    dump(path, data)
        print(f"TIME {self.operation} ({status}): {report['total_seconds']:.3f} seconds", flush=True)


def timed_run(operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            timer = RunTimer(operation)
            token = _active_timer.set(timer)
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                try:
                    # --help is not a failed experiment and must not create output.
                    if not (isinstance(error, SystemExit) and error.code in (0, None) and timer.output is None):
                        timer.finish('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed')
                except OSError:
                    pass  # Never mask the original training/evaluation failure.
                raise
            else:
                timer.finish('completed')
                return result
            finally:
                _active_timer.reset(token)
        return wrapped
    return decorate


@contextmanager
def timed_phase(name, **details):
    """Non-overlapping work blocks; uninstrumented overhead stays in total_seconds."""
    started = time.perf_counter()
    status = 'failed'
    try:
        yield
        status = 'completed'
    finally:
        timer = _active_timer.get()
        if timer is not None:
            timer.phases.append({'name': name, **details, 'status': status,
                                 'seconds': time.perf_counter() - started})


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
    timer = _active_timer.get()
    if timer is not None:
        timer.output = path
        dump(path/'timing.json', timer.report('running'))
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

# Qwen без LoRA

Основной эксперимент — **без обучения**:

```bash
python qwen/prepare.py
python qwen/evaluate.py --output runs/qwen_baseline
```

Локальный LM Studio, как в smart_export:

```bash
python qwen/evaluate.py --base-url http://localhost:1234/v1 --model ИМЯ_МОДЕЛИ --output runs/qwen_lmstudio
```

Необязательный полный SFT, если нужно сравнить обучение всех весов и LoRA:

```bash
python qwen/train.py --confirm-full-finetune --epochs 1 --output runs/qwen_full_v1
python qwen/evaluate.py --model runs/qwen_full_v1/model --output runs/qwen_full_v1_test
```

Полный SFT требует существенно больше памяти; слияние не нужно. Для сравнения используйте ту же базу и те же тестовые ID. Полное руководство — в корневом README.

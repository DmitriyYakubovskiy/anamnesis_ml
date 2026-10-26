# NER + классификатор

```bash
python ner_classifier/prepare.py
python ner_classifier/train.py --output runs/encoder_v1
python ner_classifier/evaluate.py --models runs/encoder_v1 --output runs/encoder_v1_test
```

Запускать из корня. По одной NER-модели и одному классификатору на активную колонку. `--task ner` / `--task classifier` выбирает одну задачу. `--model` меняет encoder. По умолчанию RuBERT tiny2, лимит 2048.

NER учится на координатах фрагментов независимо от значения true/false. Классификатор учится на **полном тексте анамнеза** и общем значении колонки; `null` — третий класс текущего bool. Результаты отдельных фрагментов не используются как замена общего значения. Это две параллельные задачи, не каскад и не классификатор выделенного куска.

Форматы: `*_ner.jsonl` содержит символьные сущности, `*_classifier.jsonl` — категории. Токены/BIO создаются выбранным tokenizer при обучении. Избыточно длинные записи не обрезаются; невыразимые токенами границы отражаются в alignment_report и exact span F1. Смотри корневой README.

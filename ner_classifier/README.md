# NER + классификатор: обучение с балансировкой

Запускать из корня проекта. В PowerShell команды те же:

```bash
python ner_classifier/train.py --epochs 5 --output runs/encoder_v2
python ner_classifier/evaluate.py --models runs/encoder_v2 --split validation --output runs/encoder_v2_validation
```

По одной NER-модели и одному классификатору на активную колонку. `--task ner` / `--task classifier` выбирает одну задачу. `--model` меняет encoder. По умолчанию RuBERT tiny2, лимит 2048 токенов. Старые runs не заменяются: используйте новое имя output.

## Что исправлено

- По умолчанию `--class-weighting balanced`: веса считаются **только по train** и используются в cross-entropy. Формула для присутствующего класса: `N / (K_present * count)`. Для NER считаются токены O/B/I, special/padding `-100` исключаются. Для классификатора считаются анамнезы каждого класса. У отсутствующего в train класса вес 0 и предупреждение — validation/test не используются для заполнения статистики.
- `class_weights.json` содержит имена классов, их количества, веса и `source_split=train`. `--class-weighting none` отключает веса для контрольного эксперимента.
- В конце **каждой эпохи** считаются validation F1: `eval_ner_f1` для NER и `eval_macro_f1` для классификатора. Precision/recall и дополнительные диагностические показатели тоже записываются в `training_metrics.json`.
- Лучший checkpoint выбирается по **максимальному validation F1**, затем восстанавливается перед сохранением model/ и диагностикой. При равном F1 сохраняется первый достигший этого результата checkpoint; низкий loss не подменяет F1.
- После обучения автоматически создаются `train_diagnostic_metrics.json` / `train_diagnostic_predictions.jsonl`, а также `validation_best_metrics.json` / `validation_best_predictions.jsonl`. Они описывают **лучший**, а не обязательно последний checkpoint.
- В диагностике классификатора есть `predicted_class_counts` и флаг `constant_prediction`; у NER — `no_predicted_entities`. Низкий F1 на train означает, что модель не справляется даже с обучающими примерами. Высокий train F1 и низкий validation F1 указывают на проблему переноса, но не гарантируют конкретную причину.
- Обучение загружает только `train.jsonl` и `validation.jsonl`. До обучения проверяется отсутствие общих ID, документов и одинаковых нормализованных текстов между ними.

## Проверка после выбора настроек

Оценка по умолчанию теперь использует **validation**. Итоговый тест запускайте явно, когда закончили подбор параметров:

```bash
python ner_classifier/evaluate.py --models runs/encoder_v2 --split test --output runs/encoder_v2_test
```

Можно отдельно проверить train (это диагностика запоминания, не оценка обобщения):

```bash
python ner_classifier/evaluate.py --models runs/encoder_v2 --split train --output runs/encoder_v2_train
```

В отчёте такой запуск имеет `metadata.diagnostic_only=true`. Не сравнивайте train F1 с тестовыми метриками Qwen.

## Семантика задач

NER учится на координатах фрагментов независимо от значения true/false. Классификатор учится на **полном тексте анамнеза** и общем значении колонки; null — третий класс текущего bool. Это две параллельные задачи, не каскад и не классификатор выделенного куска.

Checkpoint NER выбирается по exact character-span F1 относительно **исходных** выделений. Отдельно считается token-aligned F1: он показывает качество при расширении gold до границ токенов. Эти показатели нельзя смешивать; ни token accuracy, ни token-aligned F1 не используются вместо exact F1 при выборе checkpoint. Несовпадения границ по-прежнему сохраняются в alignment_report.json.

Данные и разбиение остаются прежними. Увеличение эпох, изменение lr и отключение весов подбираются только по validation. Балансировка устраняет перекос функции потерь, но не гарантирует улучшение на маленьком наборе. Полное описание проекта — в корневом README.

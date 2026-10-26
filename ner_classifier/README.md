# NER + классификатор

## Вариант по статье: spaCy + TF-IDF (CPU)

Для текущего небольшого датасета добавлен отдельный backend. NER обучается с нуля через spaCy, классификатор использует словные и символьные TF-IDF признаки и линейную модель. Веса готовой языковой модели не скачиваются. GPU не требуется; подходит и для Windows с AMD.

```bash
pip install -r requirements-hybrid.txt
python ner_classifier/train.py --backend hybrid --epochs 60 --output runs/encoder_article_v1
python ner_classifier/evaluate.py --models runs/encoder_article_v1 --split validation --output runs/encoder_article_v1_validation
```

`evaluate.py` определяет backend по сохранённому `run_config.json`. Используйте новое имя output для каждого эксперимента. Для этого backend достаточно `requirements-hybrid.txt`; полный стек PyTorch не нужен. `--model`, `--lr`, `--batch`, `--class-weighting`, `--max-steps` и `--max-length` относятся к transformer backend и не настраивают hybrid.

- spaCy: 60 эпох по умолчанию, minibatch от 1 до 8, dropout от 0.3 до 0.1, learning rate 0.001, gradient clipping 1. Лучшее состояние выбирается по exact-span F1 на validation; при равенстве — по F1 символов сущностей, затем меньшему числу FP, затем train loss. Exact F1 всегда главный. Если оптимизатор предоставляет усреднённые параметры, валидация и сохранение используют их.
- Классификатор: word n-grams 1–2 + char n-grams 3–5. Отрицания сохраняются. Словарь, IDF и `class_weight=balanced` обучаются только на train. По validation macro-F1 выбираются LinearSVC с C=0.25/1/4 или LogisticRegression с C=1; при равенстве — accuracy, затем первый кандидат. Все четыре результата сохраняются.
- Отдельные NER и классификатор для каждой колонки. Классификатор получает полный исходный текст, а не найденные NER фрагменты. `null` остаётся самостоятельным классом.
- В корне run находятся `validation_summary.json`, `run_config.json`, `schema.json`; внутри `<CODE>/ner` и `<CODE>/classifier` — модели, `training_metrics.json`, train/validation диагностика и предсказания. Отдельный evaluate сохраняет общий `metrics.json`, CSV и `predictions.jsonl`.
- Для обучения spaCy несовпадающие границы расширяются до токенов только в обучающей копии; изменения перечислены в `alignment_report.json`. Exact F1 всегда считается по оригинальным координатам. Символьный F1 — дополнительная диагностика, его нельзя сравнивать напрямую с token F1 RuBERT.

Результаты одного реального прогона и ограничения: [ARTICLE_EXPERIMENT.md](ARTICLE_EXPERIMENT.md). Тест не используется при обучении и выборе модели.

## Transformer / RuBERT

Запускать из корня проекта. В PowerShell команды те же:

```bash
python ner_classifier/train.py --epochs 5 --output runs/encoder_v2
python ner_classifier/evaluate.py --models runs/encoder_v2 --split validation --output runs/encoder_v2_validation
```

По одной NER-модели и одному классификатору на активную колонку. `--task ner` / `--task classifier` выбирает одну задачу. `--model` меняет encoder. По умолчанию RuBERT tiny2, лимит 2048 токенов. Старые runs не заменяются: используйте новое имя output.

## Что исправлено

- По умолчанию NER использует `sqrt`, классификатор — `balanced`: веса считаются **только по train** и используются в cross-entropy. Формула balanced для присутствующего класса: `N / (K_present * count)`; sqrt берёт квадратный корень из этого веса и смягчает перевес редких B-токенов. `--class-weighting balanced` явно возвращает прежнюю схему для обеих задач. Для NER считаются токены O/B/I, special/padding `-100` исключаются. Для классификатора считаются анамнезы каждого класса. У отсутствующего в train класса вес 0 и предупреждение — validation/test не используются для заполнения статистики.
- `class_weights.json` содержит имена классов, их количества, веса и `source_split=train`. `--class-weighting none` отключает веса для контрольного эксперимента.
- В конце **каждой эпохи** считаются validation F1: `eval_ner_f1` для NER и `eval_macro_f1` для классификатора. Precision/recall и дополнительные диагностические показатели тоже записываются в `training_metrics.json`.
- Лучший checkpoint выбирается по **максимальному validation F1**, затем восстанавливается перед сохранением model/ и диагностикой. При равном NER F1 выбирается больший entity-token F1, затем меньший validation loss; при равном macro-F1 классификатора — меньший validation loss. Это исправляет выбор первой эпохи, когда exact F1 всюду равен нулю. Никакой дополнительный показатель не может победить более высокий основной F1.
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

Checkpoint NER выбирается по exact character-span F1 относительно **исходных** выделений. Отдельно считается token-aligned F1: он показывает качество при расширении gold до границ токенов. Эти показатели нельзя смешивать; exact F1 остаётся главным критерием, entity-token F1 используется только при его равенстве; token accuracy и token-aligned span F1 не заменяют основной критерий. Несовпадения границ по-прежнему сохраняются в alignment_report.json.

Данные и разбиение остаются прежними. Увеличение эпох, изменение lr и отключение весов подбираются только по validation. Балансировка устраняет перекос функции потерь, но не гарантирует улучшение на маленьком наборе. Полное описание проекта — в корневом README.


## Время и дальнейшие улучшения

Время автоматически сохраняется в корневом `<output>/timing.json` для обучения и оценки. В `phases` отдельно перечислены обучение с валидацией, загрузка и обработка для каждой задачи. Общий формат и границы измерений описаны в корневом README. Рекомендации по улучшению с источниками: [NEXT_STEPS.md](NEXT_STEPS.md).

# Anamnesis ML: LoRA / NER + классификатор / Qwen

Отдельный Python-проект для экспорта `smart_export_dataset_creator`. Он не изменяет исходный сервис и его базу. Все команды ниже выполняются **из корня этого проекта**. Модели и медицинские тексты обрабатываются локально. Вызов LM Studio производится только при явном `--base-url`.

## Три папки

| Папка | Что делает | Подготовка | Обучение | Проверка |
|---|---|---|---|---|
| `Lora/` | Qwen + обучаемый LoRA/QLoRA-адаптер, JSON со значениями полей | `prepare.py` | `train.py` | `evaluate.py` |
| `ner_classifier/` | BIO NER по размеченным фрагментам + отдельный классификатор всего анамнеза | `prepare.py` | `train.py` | `evaluate.py` |
| `qwen/` | Исходный Qwen без дообучения, тот же промпт и тест | `prepare.py` | Не требуется; `train.py` реализует **необязательный полный SFT** | `evaluate.py` |

Общий код вынесен в `common/`, чтобы форматы, правила `null` и метрики не расходились. Папки — части одного репозитория; переносить отдельно только одну папку нельзя. Для генеративных подходов выход имеет вид `{"ARTERIAL_HYPERTENSION_FIX": true}`, `false` или `null`. Генеративные подходы здесь оцениваются по итоговым значениям, а не по координатам фрагментов.

В NER + классификатор используются **две независимые модели на каждый активный параметр**:

- NER: где в тексте упоминание, BIO `O/B-CODE/I-CODE`. И подтверждение, и отрицание являются упоминаниями. Нет упоминаний — все токены `O`.
- Классификатор: общее значение всего анамнеза, для текущего bool три класса `true / false / null`. Он видит полный текст, а не только найденные NER фрагменты; ошибки NER не вырезают контекст классификатора.

Это не каскад «выделить и классифицировать каждый фрагмент». `annotations[].result` проверяется и сохраняется в канонических данных, но отдельная модель результата каждого фрагмента не обучается. Такой выбор соответствует экспорту: общий итог и результаты нескольких упоминаний могут различаться. В NER папке получаем фрагменты и общий итог параллельно.

## Что в исходном JSON

В архив включены предоставленный `data/raw/anamneses.json` и уже подготовленные выборки. Данные не были синтетически расширены и не заменялись ответами модели.

- 65 завершённых анамнезов, 21 документ, 46 размеченных фрагментов.
- Один активный параметр `ARTERIAL_HYPERTENSION_FIX` типа bool: 36 `true`, 10 `false`, 19 `null` со статусом `no_data`.
- Отключённые параметры не участвуют в обучении; исходный JSON остаётся неизменным.
- Итоговый split: 46 train, 10 validation, 9 test. Конкретные ID, источники, распределения и SHA-256 исходника записаны в `*/data/manifest.json`.
- `document_id` и одинаковые нормализованные тексты не пересекаются между выборками. Split детерминирован при `seed=42`; выполняется поиск разбиения, близкого к 70/15/15 с покрытием классов.
- `document_id` **не является гарантированным patient_id**. Разные документы одного пациента могут попасть в разные выборки. Для полноценного исследования добавьте группировку по пациенту; на этих 65 записях результаты только предварительные.

65 примеров мало для устойчивой оценки. В тесте всего 9 записей, и количество редкого класса зависит от разбиения. Не выбирайте параметры обучения по тесту: используйте validation, test запускайте после выбора настроек. Стратификация не увеличивает количество наблюдений.

## Правила пустых значений

| В экспорте | В обучении |
|---|---|
| `state=value`, `value=false` | Значение «нет», не отсутствие информации |
| `state=value`, `value=0` | Реальный числовой ноль |
| `state=no_data`, `value=null` | Проверенное отсутствие сведений; целевой класс / JSON `null` |
| `state=unreviewed` | Запись исключается, причина в manifest |
| Старое `value=null` без `state` | Не считаем проверенным; исключается |
| Отключённая колонка | Игнорируется |

Подготовка отбирает записи с `completed`, всеми проверенными активными колонками и результатами фрагментов. Пустые NER-примеры сохраняются. Сначала делим анамнезы на выборки, затем создаём обучающие представления. Текст не обрезается и не нормализуется перед вычислением координат. Проверяются точный срез `text[start:end]`, связи, Unicode code point offsets и типы.

Для enum набор классов выводится из значений экспорта. Добавлять невстречавшиеся варианты лучше через явную схему до эксперимента. Для int классификатор рассматривает наблюдавшиеся числа как категории и не предсказывает новые числовые значения: для широкого диапазона нужен отдельный регрессионный подход. LoRA/Qwen возвращают типизированный int.

## Установка

Нужен **Python 3.11 или 3.12**. В Ubuntu, где по умолчанию Python 3.14, создайте окружение отдельно на 3.12; эти зависимости не проверялись на 3.14. Для Qwen SFT рекомендуется NVIDIA GPU; для маленького RuBERT возможен CPU. Конкретный расход VRAM зависит от длины текста и batch, не только от размера весов.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

В PowerShell активация: `.venv\Scripts\Activate.ps1`. Остальные команды `python ...` те же.

Для CPU можно **до** установки остальных требований установить CPU-сборку:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Для GPU поставьте CUDA-сборку PyTorch, совместимую с драйвером, по официальной инструкции PyTorch. Например, для совместимого драйвера CUDA 12.8: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`, затем остальные requirements. Проверьте:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

QLoRA дополнительно требует:

```bash
pip install -r requirements-4bit.txt
```

По умолчанию используется `Qwen/Qwen3-4B-Instruct-2507`: конкретная non-thinking модель, а не автоматически меняющийся `latest`. Параметром `--model` можно выбрать другую instruct-модель. Не подставляйте GGUF в Transformers training. При сравнении LoRA и Qwen без обучения используйте одну и ту же базу, revision и одинаковый промпт. Желательно задать HF commit через `--revision`; фактический commit записывается в конфигурацию обучения. `trust_remote_code` не включается.

## 1. Подготовить все три варианта

Уже сделано для приложенного JSON; повторная подготовка нужна после изменения исходника:

```bash
python prepare_all.py --input data/raw/anamneses.json --seed 42
```

Или отдельно:

```bash
python Lora/prepare.py --input data/raw/anamneses.json
python ner_classifier/prepare.py --input data/raw/anamneses.json
python qwen/prepare.py --input data/raw/anamneses.json
```

В каждой `data/`: канонические `train/validation/test.jsonl`, `schema.json`, `manifest.json`. В Lora/Qwen дополнительно `*_chat.jsonl` с system/user/assistant. В NER дополнительно `*_ner.jsonl` с символьными сущностями и `*_classifier.jsonl` с категориями. BIO создаётся выбранным tokenizer во время обучения — позиции токенов зависят от конкретной модели.

## 2. Базовый Qwen: начать с него

```bash
python qwen/evaluate.py --output runs/qwen_baseline
```

Ничего не обучается. Скачиваются исходные веса и выполняется генерация на тесте. Для короткой итерации по промпту используйте `--split validation --output runs/qwen_validation`.

Как в `smart_export`, можно использовать уже запущенный LM Studio:

```bash
python qwen/evaluate.py --base-url http://localhost:1234/v1 --model ИМЯ_ЗАГРУЖЕННОЙ_МОДЕЛИ --output runs/qwen_lmstudio
```

Это только оценка уже загруженной серверной модели; версию/квантование в LM Studio зафиксируйте самостоятельно. Ключ при необходимости задаётся переменной `LM_API_KEY`. В сеть отправляется текст только при явном `--base-url`; укажите адрес своего сервера. Ошибки HTTP/JSON не пропускаются из знаменателя метрик. Невалидный JSON не ремонтируется и не превращается в успешный `null`.

## 3. LoRA / QLoRA

Обычная LoRA:

```bash
python Lora/train.py --epochs 3 --output runs/lora_v1
python Lora/evaluate.py --adapter runs/lora_v1/model --output runs/lora_v1_test
```

При ограниченной VRAM — QLoRA (Linux/WSL, NVIDIA CUDA, установлен bitsandbytes):

```bash
python Lora/train.py --four-bit --epochs 3 --output runs/qlora_v1
python Lora/evaluate.py --adapter runs/qlora_v1/model --four-bit --output runs/qlora_v1_test
```

Модель учится только на ответе ассистента: prompt labels = `-100`; padding также не входит в loss. Checkpoint выбирается по validation loss, test в обучение не подмешивается. Значения по умолчанию: rank 16, alpha 32, dropout 0.05, `all-linear`, learning rate `2e-4`, batch 1, gradient accumulation 8, max length 4096. Изменяются аргументами CLI. Ответы и длинные тексты не обрезаются молча: превышение лимита вызывает явную ошибку.

Для DeepSeek можно задать подходящую базу через `--model`, если она поддерживается указанной версией Transformers и имеет chat template. Но обучение на DeepSeek создаёт **DeepSeek-адаптер**, а не Qwen-адаптер. Изученный репозиторий не содержит кода, достаточного для точного воспроизведения твоего прежнего DeepSeek-обучения.

## 4. NER + классификатор

Добавлен вариант по приложенной статье, работающий на CPU: **spaCy NER + TF-IDF / линейный классификатор**. На текущих 46 train / 10 validation получены exact NER F1 **0.20** и classifier macro-F1 **0.687**. Это предварительный результат на validation; тест для выбора моделей не запускался.

```bash
pip install -r requirements-hybrid.txt
python ner_classifier/train.py --backend hybrid --epochs 60 --output runs/encoder_article_v1
python ner_classifier/evaluate.py --models runs/encoder_article_v1 --split validation --output runs/encoder_article_v1_validation
```

Предобученные веса для hybrid не скачиваются. Подробности, отчёты и команды итогового теста — в [ner_classifier/README.md](ner_classifier/README.md), описание эксперимента — в [ARTICLE_EXPERIMENT.md](ner_classifier/ARTICLE_EXPERIMENT.md).


По умолчанию небольшой русский `cointegrated/rubert-tiny2` с поддержкой до 2048 токенов:

```bash
python ner_classifier/train.py --epochs 5 --output runs/encoder_v2
python ner_classifier/evaluate.py --models runs/encoder_v2 --split validation --output runs/encoder_v2_validation
```

Веса O/B/I и классов параметра вычисляются только по **train** (по умолчанию NER: `sqrt`, классификатор: `balanced`; явный `--class-weighting balanced` возвращает прежние веса) и сохраняются в `class_weights.json`. На каждой валидации считаются exact NER F1 и macro-F1 классификатора. Лучший checkpoint выбирается по максимальному соответствующему **validation F1** и восстанавливается перед сохранением. LoRA/Qwen продолжают использовать свой прежний выбор по validation loss.

После обучения автоматически появляются `train_diagnostic_metrics.json` / `train_diagnostic_predictions.jsonl` и `validation_best_metrics.json` / `validation_best_predictions.jsonl` для каждой задачи. В них видны предсказания лучшего checkpoint на train и validation; диагностический train F1 не является оценкой обобщения. Подробности формулы весов и метрик — в `ner_classifier/README.md`.

`ner_classifier/evaluate.py` по умолчанию использует validation, поддерживает `--split train` для диагностики. Когда подбор настроек закончен, запустите итоговый тест явно:

```bash
python ner_classifier/evaluate.py --models runs/encoder_v2 --split test --output runs/encoder_v2_test
```

Можно обучить / проверить только одну задачу: `--task ner` или `--task classifier`. Для другой модели передайте `--model`; `--max-length` не должен превышать её лимит.

На твоём JSON проверено tokenizer RuBERT: максимум 612 токенов, все записи помещаются в лимит 2048. **У 11 из 46 фрагментов символьные границы не совпадают с границами токенов.** В обучении BIO помечаются пересекающиеся токены; факты расширения записываются в `alignment_report.json`. Исходная разметка не меняется. На оценке exact span F1 сравнивается с оригинальными координатами, поэтому такие границы могут снижать его даже при верном распознавании смысла. Полезно согласовать правила выделения перед расширением датасета. Пересечения одной метки или два фрагмента внутри одного токена отклоняются; пересечения разных меток поддерживаются отдельными NER-моделями.

## 5. Объединить LoRA + Qwen

```bash
python Lora/merge.py --adapter runs/lora_v1/model --output models/qwen_merged
python qwen/evaluate.py --model models/qwen_merged --output runs/qwen_merged_test
```

Скрипт сам читает исходную базу и revision из `adapter_config.json`, загружает её **без 4-битного квантования**, вызывает `merge_and_unload(safe_merge=True)` и сохраняет самостоятельную модель с tokenizer. Потребуется память под полные веса. После QLoRA тоже объединяйте адаптер с исходной неквантованной базой; квантование для развёртывания делается уже после merge.

**LoRA от DeepSeek нельзя слить с Qwen.** Нужна именно та базовая архитектура и исходные веса, на которых обучали адаптер. Одинаковый размер матриц сам по себе недостаточен. Для проверки equivalence сравнивайте адаптер и merged модель в одной точности; 4-битный inference может немного отличаться.

Для LM Studio папку HF Safetensors нужно конвертировать в GGUF, если приложение/загрузчик не принимает эту HF-модель. В актуальном checkout `llama.cpp` после установки его Python-зависимостей:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py models/qwen_merged --outfile models/qwen_merged-f16.gguf --outtype f16
/path/to/llama.cpp/build/bin/llama-quantize models/qwen_merged-f16.gguf models/qwen_merged-Q4_K_M.gguf Q4_K_M
```

Поддержка выбранной архитектуры зависит от версии llama.cpp; этот шаг здесь не запускался. Выберите полученный GGUF в LM Studio и проверьте chat template. Нельзя просто переименовать `.safetensors` в `.gguf`.

## 6. Необязательный полный SFT Qwen

Для базового эксперимента не нужен. Это изменение всех весов, а не LoRA; заметно дороже по памяти и легче переобучается на 46 train-примерах:

```bash
python qwen/train.py --confirm-full-finetune --epochs 1 --output runs/qwen_full_v1
python qwen/evaluate.py --model runs/qwen_full_v1/model --output runs/qwen_full_v1_test
```

Слияние не требуется: итог — полная модель. Для него специально нет режима `--four-bit`.

## Метрики и результаты

Каждая `evaluate.py` сохраняет:

- `metrics.json`: accuracy, macro-F1, exact match, invalid rate; распределения/ID выборок, параметры и версии библиотек.
- `field_metrics.csv`: precision/recall/F1/support каждого класса каждого поля; UTF-8 BOM, разделитель `;`.
- `predictions.jsonl`: предсказание каждого анамнеза выбранной выборки, исходный ответ LLM или NER-фрагменты, ошибки. У генеративных моделей также время ответа.
- `errors.json`: невалидный JSON, HTTP/другие ошибки. Ошибки считаются ошибками, не удаляются из теста.
- NER: exact character-span precision/recall/F1, entity-token detection F1, token accuracy, количество непредставимых границ и `alignment_report.json`. При `--task ner` CSV классификатора не создаётся.

Macro-F1 считается по фиксированному словарю классов, включая `null`; отсутствующий класс имеет support 0 и F1 0. Это явно видно в CSV. `invalid_rate` отличает отсутствие сведений в тексте от сломанного ответа модели. Пустой/пропущенный результат не становится `null`. NER оценивается отдельно: его span F1 нельзя напрямую сравнивать с F1 классификатора.

Обучение сохраняет `training_metrics.json`, историю loss и validation loss, параметры, версии, `run_config.json` и `model/`. Папка output должна быть новой/пустой: случайной перезаписи эксперимента нет. Для повторного запуска выберите новое имя.

Сравнить значения на одном тесте:

```bash
python compare.py runs/qwen_baseline/metrics.json runs/lora_v1_test/metrics.json runs/encoder_v2_test/metrics.json --output reports/comparison.json
```

Скрипт проверяет совпадение исходного SHA-256, split и тестовых ID. Эксперименты с разными наборами не сравниваются автоматически. Тип/квантование LLM фиксируйте в названии запуска; сервер LM Studio и локальный Transformers не гарантируют побитно одинаковые результаты.

## Проверки проекта

```bash
python -m pytest -q tests/test_core.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/smoke_training.py
```

`smoke_training.py` создаёт временные **случайные крошечные** Qwen3/BERT и синтетические записи. Проверяет весь маршрут обучения/оценки, сохранение файлов, adapter merge и совпадение logits. Он не скачивает медицинские модели, не проверяет качество Qwen4B и не выдаёт свои метрики за результаты на твоём датасете. Выполненные проверки описаны в `reports/VALIDATION.md`.

## Подготовка репозитория

Архив содержит твой медицинский JSON и производные train/test в игнорируемых каталогах. `.gitignore` исключает `data/raw`, подготовленные данные, `runs`, `models`, отчёты с примерами и окружение. Перед любым будущим `git add` проверь `git status --short --ignored`: публиковать нужно код, а не эти данные. Новый удалённый репозиторий в этом запросе не создавался.

## Что изучено в smart_export

Проверена ветка `main`, commit `19fb2899383ec27019d7c3f3d1fe09859f7257df`:

- `smart_processing_module/main.py`: преобразование анамнезов через локальную LLM с валидацией JSON.
- `lm_studio.py`: OpenAI-compatible `/chat/completions`, очистка/ремонт JSON.
- `config.json`: имя `qwen3_4b_anamnesis`.
- `metrics_evaluator.py`: отдельные отчёты метрик по эталонному JSON.

В дереве main кода обучения LoRA/DeepSeek и notebook с этим обучением не найдено. Поэтому это новая реализация обучения; от прежнего контура взята идея локального inference, строгой схемы и файловых отчётов. JSON repair при подсчёте новых метрик намеренно не применяется.

## Официальные источники

- Репозиторий: https://github.com/DmitriyYakubovskiy/smart_export/tree/19fb2899383ec27019d7c3f3d1fe09859f7257df/smart_processing_module
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Trainer 4.57.1: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
- PEFT quantization: https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization
- PEFT merge: https://huggingface.co/docs/peft/en/package_reference/lora
- RuBERT tiny2: https://huggingface.co/cointegrated/rubert-tiny2
- PyTorch installation: https://pytorch.org/get-started/locally/
- llama.cpp conversion: https://github.com/ggml-org/llama.cpp


## Время обучения и оценки

Все команды `train.py` и `evaluate.py` в Lora, qwen и ner_classifier (оба backend) автоматически сохраняют `<output>/timing.json` и печатают `TIME ... seconds`.

- `total_seconds`: полное wall-clock время тела команды — разбор аргументов, ленивые импорты, чтение данных, загрузка/скачивание модели, работа, диагностика и экспорт результатов. Запуск интерпретатора и импорты модулей до входа в функцию, а также финальная запись самого отчёта времени не включены.
- `phases`: `fit_and_validation` — обучение с валидацией и выбором checkpoint; при оценке `model_load` — загрузка модели, `inference` — обработка выборки с постобработкой. Для NER и классификатора этапы имеют `task` и `column`. У TF-IDF время обучения охватывает все четыре проверяемых кандидата. Этапы не покрывают весь overhead; их сумма может быть меньше total.
- Время указано в секундах; `started_at` / `finished_at` — ISO 8601 UTC; `status` — running/completed/failed/cancelled. При обычном исключении после принятия output сохраняется failed; при Ctrl+C — cancelled. При принудительном завершении процесса может остаться running. При ошибке до создания output файл не создаётся; существующий run не перезаписывается.
- В evaluate тот же отчёт дублируется в `metrics.json → metadata.timing`; в обучении LoRA/Qwen — в корневом `training_metrics.json → timing`, у hybrid — в `validation_summary.json → timing`. У transformer NER общий отчёт находится в корне run в `timing.json`, отдельные fit-этапы — в его phases.

Прежние поля `metadata.seconds`, `train_runtime` и время отдельных ответов LLM сохранены для совместимости; их границы измерения отличаются. Для новых сравнений используйте одинаковые поля из timing.json, одинаковую выборку, оборудование и режим кэша. `total_seconds` при первом скачивании нельзя напрямую сравнивать с запуском из кэша. Для API model_load измеряет создание клиента; загрузка модели на сервере отдельно не измеряется, а inference включает сетевую задержку. Эти замеры являются временем реального запуска, а не чистым GPU-бенчмарком.

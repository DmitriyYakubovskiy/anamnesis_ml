# LoRA / QLoRA

Запуск из корня проекта, после установки requirements:

```bash
python Lora/prepare.py
python Lora/train.py --output runs/lora_v1
python Lora/evaluate.py --adapter runs/lora_v1/model --output runs/lora_v1_test
python Lora/merge.py --adapter runs/lora_v1/model --output models/qwen_merged
```

Для 4-bit установите `requirements-4bit.txt`, добавьте `--four-bit` к train/evaluate. Merge всегда использует исходную неквантованную базу. LoRA от DeepSeek с Qwen несовместима.

Промпт замаскирован в loss; учится только JSON ответа. Данные `data/*_chat.jsonl` доступны для просмотра, train использует эквивалентные канонические записи и общую функцию промпта. Полные команды, метрики, ограничения и правила null — в корневом README.


## Проверка после обучения и disk offload

Адаптер можно проверить без merge и повторного обучения:

```powershell
python Lora/evaluate.py --adapter runs/lora_v1/model --split validation --output runs/lora_v1_validation_retry
```

При `device_map=auto` часть весов может размещаться на диске, если доступной памяти недостаточно. Evaluate автоматически использует `<output>/offload` и передаёт эту папку как загрузчику базовой модели, так и PEFT при подключении LoRA. Это исправляет `ValueError: We need an offload_dir to dispatch this model`. Выгрузка на диск может заметно замедлить генерацию; требуется свободное место. Обученный адаптер не изменяется.

Папку можно задать явно, например на SSD. Используйте отдельную папку для каждого запуска:

```powershell
python Lora/evaluate.py --adapter runs/lora_v1/model --split validation --output runs/lora_v1_validation_ssd --offload-folder D:/ml_offload/lora_validation
```

`--device-map cpu` размещает все веса в RAM и требует достаточно оперативной памяти для модели и генерации. Значение по умолчанию — `auto`. `--four-bit` нельзя совмещать с `--device-map cpu`.

После неудачной проверки используйте новое имя output. Метрики и время сохраняются в `metrics.json` и `timing.json`. Для `qwen/evaluate.py` доступны те же параметры загрузки. При обращении к API выгрузкой управляет сервер.

Регрессионный тест: `python -m pytest tests/test_lora_loading.py -q`. Он использует случайную маленькую Qwen и локальный LoRA, принудительно направляет один слой на диск при повторном распределении PEFT и проверяет совпадение logits с RAM-вариантом; качество модели этот тест не оценивает.

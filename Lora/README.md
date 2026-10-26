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

"""Strict typed value metrics. Invalid/missing prediction is never treated as null."""
from collections import Counter
import csv
import json
from pathlib import Path
from common.data import dump, key, valid_value

INVALID = '__INVALID_PREDICTION__'


def parse_answer(raw, schema):
    # No JSON repair: malformed outputs must count as failures.
    obj = json.loads(raw.strip())
    if not isinstance(obj, dict) or set(obj) != {c['code'] for c in schema}:
        raise ValueError('Response keys do not match schema')
    for col in schema:
        if not valid_value(col, obj[col['code']]):
            raise ValueError('Invalid response type/value for ' + col['code'])
    return obj


def classification_report(gold, predicted, classes):
    assert len(gold) == len(predicted)
    classes = list(dict.fromkeys(classes))
    per_class = {}
    for cls in classes:
        tp = sum(g == cls and p == cls for g, p in zip(gold, predicted))
        fp = sum(g != cls and p == cls for g, p in zip(gold, predicted))
        fn = sum(g == cls and p != cls for g, p in zip(gold, predicted))
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': gold.count(cls)}
    return {'n': len(gold), 'accuracy': sum(g == p for g, p in zip(gold, predicted))/len(gold) if gold else 0,
            'macro_f1': sum(c['f1'] for c in per_class.values())/len(classes) if classes else 0,
            'balanced_accuracy_present_classes': sum(c['recall'] for c in per_class.values() if c['support'])/sum(bool(c['support']) for c in per_class.values()) if gold else 0,
            'per_class': per_class,
            'confusion': {g: dict(Counter(p for gg, p in zip(gold, predicted) if gg == g)) for g in classes}}


def value_metrics(rows, predictions, schema):
    by_id = {r['id']: r for r in predictions}
    fields, exact, invalid = {}, 0, 0
    for r in rows:
        pred = by_id.get(r['id'], {})
        obj = pred.get('prediction')
        is_valid = isinstance(obj, dict) and set(obj) == {c['code'] for c in schema} and all(valid_value(c, obj[c['code']]) for c in schema)
        invalid += not is_valid
        exact += is_valid and all(key(r['target'][c['code']]) == key(obj[c['code']]) for c in schema)
    for col in schema:
        code = col['code']
        gold = [key(r['target'][code]) for r in rows]
        pred = []
        for r in rows:
            obj = by_id.get(r['id'], {}).get('prediction')
            # Whole malformed object is invalid, consistent with exact schema contract.
            valid = isinstance(obj, dict) and set(obj) == {c['code'] for c in schema} and all(valid_value(c, obj[c['code']]) for c in schema)
            pred.append(key(obj[code]) if valid else INVALID)
        fields[code] = classification_report(gold, pred, [key(v) for v in col['values']])
    return {'n': len(rows), 'exact_match': exact/len(rows) if rows else 0,
            'invalid_count': invalid, 'invalid_rate': invalid/len(rows) if rows else 0,
            'macro_f1': sum(f['macro_f1'] for f in fields.values())/len(fields), 'fields': fields}


def span_metrics(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    tp, fp, fn = len(gold & predicted), len(predicted-gold), len(gold-predicted)
    p, r = tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': 2*p*r/(p+r) if p+r else 0,
            'empty_gold_and_prediction': not gold and not predicted}


def save_metrics(out, metrics, metadata):
    out = Path(out)
    dump(out/'metrics.json', {**metrics, 'metadata': metadata})
    if 'fields' in metrics:
        with (out/'field_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['field', 'class_json', 'precision', 'recall', 'f1', 'support'])
            for code, report in metrics['fields'].items():
                for cls, values in report['per_class'].items():
                    writer.writerow([code, cls, values['precision'], values['recall'], values['f1'], values['support']])

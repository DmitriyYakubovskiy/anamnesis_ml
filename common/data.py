"""No model dependencies: deterministic document-group splits and strict export validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def valid_value(col, value):
    if value is None:
        return True
    kind = col['data_type']
    return ((kind == 'bool' and type(value) is bool)
            or (kind == 'int' and type(value) is int and abs(value) <= 9007199254740991)
            or (kind == 'enum' and type(value) is str and value in col['values']))


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty array exported from dataset creator')
    schema = {}
    for r in rows:
        for c in r['columns']:
            if not c['active']:
                continue
            code = c['code']
            if not re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', code):
                raise ValueError('Unsafe/invalid column code')
            spec = {'code': code, 'name': c['name'], 'data_type': c['data_type'], 'values': []}
            if code in schema and any(schema[code][f] != spec[f] for f in ('name', 'data_type')):
                raise ValueError(f'Conflicting column schema: {code}')
            schema.setdefault(code, spec)
            v = c['value']
            if v is not None and key(v) not in {key(x) for x in schema[code]['values']}:
                schema[code]['values'].append(v)
    if not schema:
        raise ValueError('No active columns')
    for col in schema.values():
        if col['data_type'] not in {'int', 'bool', 'enum'}:
            raise ValueError('Unsupported column type')
        if col['data_type'] == 'bool':
            col['values'] = [True, False]
        col['values'] = sorted(col['values'], key=key)
        col['values'].append(None)
    kept, excluded, seen = [], [], set()
    for r in rows:
        rid = r['id']
        if rid in seen:
            raise ValueError(f'Duplicate ID: {rid}')
        seen.add(rid)
        if not r.get('document_id') or not isinstance(r.get('text'), str) or not r['text'].strip():
            raise ValueError(f'Missing source or text: {rid}')
        if r.get('offset_unit', 'unicode_codepoint') != 'unicode_codepoint' or r.get('end_exclusive', True) is not True:
            raise ValueError('Expected Unicode code point offsets with exclusive end')
        if not r.get('completed'):
            excluded.append({'id': rid, 'reason': 'annotation_not_completed'})
            continue
        cols = {c['code']: c for c in r['columns'] if c['active']}
        if set(cols) != set(schema):
            raise ValueError(f'Inconsistent active column set: {rid}')
        target, pending = {}, False
        for code, c in cols.items():
            # Legacy null has no evidence of review: never silently becomes no_data.
            state = c.get('state', 'unreviewed' if c['value'] is None else 'value')
            if state == 'unreviewed':
                if c['value'] is not None:
                    raise ValueError(f'Unreviewed value is not null: {rid}/{code}')
                pending = True
            elif state == 'no_data':
                if c['value'] is not None:
                    raise ValueError(f'no_data must be null: {rid}/{code}')
            elif state != 'value' or c['value'] is None:
                raise ValueError(f'Invalid value state: {rid}/{code}')
            if not valid_value(schema[code], c['value']):
                raise ValueError(f'Invalid value type: {rid}/{code}')
            target[code] = c['value']
        if pending:
            excluded.append({'id': rid, 'reason': 'unreviewed_active_column'})
            continue
        spans = []
        for a in r.get('annotations', []):
            code = a['label']
            if code not in schema:
                continue
            s, e = a['start'], a['end']
            if type(s) is not int or type(e) is not int or not (0 <= s < e <= len(r['text'])) or r['text'][s:e] != a['text']:
                raise ValueError(f'Invalid span offsets/text: {rid}')
            if a.get('history_id', rid) != rid or a.get('document_id', r['document_id']) != r['document_id']:
                raise ValueError(f'Invalid span source: {rid}')
            state = a.get('result_state', 'unreviewed' if a.get('result') is None else 'value')
            if state == 'unreviewed':
                pending = True
            elif state not in {'value', 'no_data'} or (state == 'no_data') != (a.get('result') is None):
                raise ValueError(f'Invalid span result state: {rid}')
            if not valid_value(schema[code], a.get('result')):
                raise ValueError(f'Invalid span result type: {rid}')
            spans.append({'label': code, 'start': s, 'end': e, 'text': a['text'], 'result': a.get('result')})
        if pending:
            excluded.append({'id': rid, 'reason': 'unreviewed_span_result'})
            continue
        for code in schema:
            ss = sorted((a['start'], a['end']) for a in spans if a['label'] == code)
            if any(s2 < e1 for (s1, e1), (s2, e2) in zip(ss, ss[1:])):
                raise ValueError(f'Overlapping same-label spans cannot be represented in BIO: {rid}/{code}')
        kept.append({'id': rid, 'document_id': r['document_id'], 'text': r['text'], 'target': target, 'spans': spans})
    if len(kept) < 3:
        raise ValueError('Need at least 3 reviewed histories')
    return kept, list(sorted(schema.values(), key=lambda c: c['code'])), excluded


def split_groups(items, seed=42):
    items = sorted(items, key=lambda r: r['id'])
    # Merge document groups with identical normalized text to prevent duplicate leakage.
    parent = {r['document_id']: r['document_id'] for r in items}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_text = {}
    for r in items:
        text = normalize_text(r['text'])
        if text in by_text:
            parent[find(r['document_id'])] = find(by_text[text])
        by_text[text] = r['document_id']
    groups = {}
    for r in items:
        groups.setdefault(find(r['document_id']), []).append(r)
    ids = sorted(groups)
    if len(ids) < 3:
        raise ValueError('Need at least 3 independent document/duplicate groups')
    n_eval = max(1, round(len(ids) * .15))
    counts = Counter((c, key(v)) for r in items for c, v in r['target'].items())
    rng, best = random.Random(seed), None
    for _ in range(1500):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        assignment = {'validation': shuffled[:n_eval], 'test': shuffled[n_eval:2*n_eval], 'train': shuffled[2*n_eval:]}
        parts = {s: [r for gid in gg for r in groups[gid]] for s, gg in assignment.items()}
        score = 0
        for s, fraction in [('train', .7), ('validation', .15), ('test', .15)]:
            observed = Counter((c, key(v)) for r in parts[s] for c, v in r['target'].items())
            score += abs(len(parts[s])/len(items) - fraction)
            score += sum(abs(observed[k] / n - fraction) for k, n in counts.items())
            score += sum((100 if s == 'train' else 10) for k in counts if not observed[k])
        if best is None or score < best[0]:
            best = score, parts
    return {s: sorted(rr, key=lambda r: r['id']) for s, rr in best[1].items()}


def prepare(input_path, out, seed=42):
    out = Path(out)
    rows = load(input_path)
    items, schema, excluded = validate(rows)
    parts = split_groups(items, seed)
    manifest = {'seed': seed, 'source_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                'schema': schema, 'grouping': 'document_id + normalized identical text',
                'excluded': excluded, 'splits': {s: [r['id'] for r in rr] for s, rr in parts.items()},
                'document_ids': {s: sorted({r['document_id'] for r in rr}) for s, rr in parts.items()},
                'distribution': {s: {c['code']: dict(Counter(key(r['target'][c['code']]) for r in rr)) for c in schema} for s, rr in parts.items()}}
    manifest['warnings'] = ['document_id is not patient_id: different documents of one patient may still leak; supply patient grouping before a clinical study.',
                            'This small dataset is for pipeline debugging, not a reliable estimate of clinical performance.',
                            'Enum/int classification vocabulary is inferred from export values; integers are categorical, not regression.']
    dump(out/'manifest.json', manifest)
    dump(out/'schema.json', schema)
    for s, rr in parts.items():
        jsonl(out/f'{s}.jsonl', rr)
    return manifest, parts



def export_task_formats(folder, out, manifest, parts):
    out = Path(out)
    if folder in {'Lora', 'qwen'}:
        from common.prompts import messages
        for split, rows in parts.items():
            jsonl(out/f'{split}_chat.jsonl', [{'id': r['id'], 'messages': messages(r['text'], manifest['schema']) + [{'role': 'assistant', 'content': key(r['target'])}]} for r in rows])
    else:
        for split, rows in parts.items():
            jsonl(out/f'{split}_ner.jsonl', [{'id': r['id'], 'document_id': r['document_id'], 'text': r['text'], 'entities': [{'label': a['label'], 'start': a['start'], 'end': a['end'], 'text': a['text']} for a in r['spans']]} for r in rows])
            jsonl(out/f'{split}_classifier.jsonl', [{'id': r['id'], 'document_id': r['document_id'], 'text': r['text'], 'labels': {code: key(value) for code, value in r['target'].items()}} for r in rows])


def prepare_cli(folder):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT/'data/raw/anamneses.json'))
    parser.add_argument('--out', default=str(ROOT/folder/'data'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest, parts = prepare(args.input, args.out, args.seed)
    export_task_formats(folder, args.out, manifest, parts)
    print(json.dumps({s: len(rr) for s, rr in parts.items()}))

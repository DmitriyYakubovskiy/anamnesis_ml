"""Train-only class balancing and task metrics shared by validation and diagnostics."""
from collections import Counter
from common.encoder import decode_bio
from common.metrics import classification_report, span_metrics
from common.data import key, normalize_text



def audit_training_parts(parts):
    for split in ['train','validation']:
        if not parts[split]:
            raise ValueError(f'{split} split is empty')
        ids = [r['id'] for r in parts[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {split}')
    for field,transform in [('id',lambda x:x),('document_id',lambda x:x),('text',normalize_text)]:
        left = {transform(r[field]) for r in parts['train']}
        right = {transform(r[field]) for r in parts['validation']}
        if left & right:
            raise ValueError(f'Train/validation leakage in {field}')


def class_balance(examples, label_names, task, enabled=True):
    """Inverse-frequency weights N/(K_present * count), excluding ignored tokens.

    Only pass train examples. Absent classes get zero weight and are reported;
    they are never given fabricated observations from validation/test.
    """
    counts = Counter()
    for example in examples:
        labels = example['labels'] if task == 'ner' else [example['labels']]
        for label in labels:
            if label == -100:
                continue
            if not 0 <= label < len(label_names):
                raise ValueError('Invalid class index in train data')
            counts[int(label)] += 1
    total = sum(counts.values())
    if not total:
        raise ValueError('Train data has no supervised targets')
    present = len(counts)
    weights = [total/(present*counts[i]) if counts[i] else 0.0 for i in range(len(label_names))]
    if not enabled:
        weights = [1.0]*len(label_names)
    return {'method': 'balanced_inverse_frequency' if enabled else 'none', 'source_split': 'train',
            'ignored_label': -100, 'total_targets': total,
            'counts': {name: counts[i] for i,name in enumerate(label_names)},
            'weights': {name: weights[i] for i,name in enumerate(label_names)},
            'missing_classes': [name for i,name in enumerate(label_names) if not counts[i]]}, weights


def summarize_predictions(predicted_ids, label_ids, rows, offsets, column, task):
    """Use original gold character spans for checkpoint selection, not expanded BIO gold."""
    code = column['code']
    predictions = []
    if task == 'classifier':
        classes = [key(value) for value in column['values']]
        gold, pred = [], []
        for row,index in zip(rows,predicted_ids):
            index = int(index)
            gold.append(key(row['target'][code]))
            pred.append(classes[index])
            predictions.append({'id':row['id'], 'gold':row['target'][code], 'prediction':column['values'][index]})
        report = classification_report(gold,pred,classes)
        report['predicted_class_counts'] = dict(Counter(pred))
        report['constant_prediction'] = len(set(pred))==1 and len(set(gold))>1
        flat = {name:report[name] for name in ['accuracy','macro_f1','balanced_accuracy_present_classes']}
        for i,name in enumerate(classes):
            for metric in ['precision','recall','f1','support']:
                flat[f'class_{i}_{metric}'] = report['per_class'][name][metric]
        return flat, report, predictions
    gold_spans, pred_spans, aligned_gold = [], [], []
    token_gold, token_pred = [], []
    for row,pred,gold,off in zip(rows,predicted_ids,label_ids,offsets):
        pred = [int(x) for x in pred[:len(off)]]
        gold = [int(x) for x in gold[:len(off)]]
        gold_spans.extend((row['id'],code,a['start'],a['end']) for a in row['spans'] if a['label']==code)
        predicted_spans = decode_bio(pred,off)
        pred_spans.extend((row['id'],code,s,e) for s,e in predicted_spans)
        aligned_gold.extend((row['id'],code,s,e) for s,e in decode_bio(gold,off))
        for g,p in zip(gold,pred):
            if g != -100:
                token_gold.append(g);token_pred.append(p)
        predictions.append({'id':row['id'], 'gold_spans':[a for a in row['spans'] if a['label']==code],
                            'predicted_spans':[{'start':s,'end':e,'text':row['text'][s:e]} for s,e in predicted_spans]})
    exact = span_metrics(gold_spans,pred_spans)
    aligned = span_metrics(aligned_gold,pred_spans)
    token = span_metrics([i for i,g in enumerate(token_gold) if g>0],[i for i,p in enumerate(token_pred) if p>0])
    flat = {'ner_f1':exact['f1'], 'ner_precision':exact['precision'], 'ner_recall':exact['recall'],
            'ner_tp':exact['tp'], 'ner_fp':exact['fp'], 'ner_fn':exact['fn'],
            'ner_token_aligned_f1':aligned['f1'], 'ner_entity_token_f1':token['f1'],
            'token_accuracy':sum(g==p for g,p in zip(token_gold,token_pred))/len(token_gold) if token_gold else 0}
    return flat, {'exact_spans':exact,'token_aligned_spans':aligned,'entity_tokens':token,
                  'no_predicted_entities':bool(gold_spans) and not pred_spans,
                  'token_accuracy':flat['token_accuracy'],'n':len(rows)}, predictions


def metric_function(rows, offsets, column, task):
    def compute(evaluation):
        predicted = evaluation.predictions
        if isinstance(predicted,tuple):
            predicted = predicted[0]
        # Trainer stores argmax IDs (preprocess_logits); accept raw logits too.
        expected_ndim = 2 if task == 'ner' else 1
        if predicted.ndim == expected_ndim + 1:
            predicted = predicted.argmax(-1)
        return summarize_predictions(predicted,evaluation.label_ids,rows,offsets,column,task)[0]
    return compute


def argmax_logits(logits, labels):
    if isinstance(logits,tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def weighted_trainer_class():
    # Lazy dependency: preparation and pure metric tests do not require torch.
    import torch
    from transformers import Trainer
    class WeightedTrainer(Trainer):
        def __init__(self,*args,class_weights,**kwargs):
            super().__init__(*args,**kwargs)
            self.class_weights = torch.tensor(class_weights,dtype=torch.float32)
            self.model_accepts_loss_kwargs = False
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels = inputs['labels']
            outputs = model(**{k:v for k,v in inputs.items() if k != 'labels'})
            logits = outputs.logits
            weights = self.class_weights.to(logits.device)
            # Explicit numerator/denominator also handles all-ignored batches safely.
            flat_labels = labels.reshape(-1)
            valid = flat_labels != -100
            if not valid.any():
                loss = logits.float().sum()*0
            else:
                losses = torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),flat_labels,
                    weight=weights,ignore_index=-100,reduction='sum')
                denominator = weights[flat_labels[valid]].sum()
                loss = losses/denominator.clamp_min(torch.finfo(torch.float32).eps)
            return (loss,outputs) if return_outputs else loss
    return WeightedTrainer

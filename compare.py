"""Compare held-out metrics only when source and test IDs are identical."""
import argparse
from common.data import load,dump
p=argparse.ArgumentParser()
p.add_argument('reports',nargs='+',help='Paths to metrics.json')
p.add_argument('--output',default='reports/comparison.json')
a=p.parse_args();reports=[load(f) for f in a.reports]
first=reports[0]['metadata']
for r in reports:
    meta=r['metadata']
    if meta['split'] != first['split'] or meta['manifest']['source_sha256'] != first['manifest']['source_sha256'] or meta['manifest']['splits'][meta['split']] != first['manifest']['splits'][first['split']]:
        raise ValueError('Cannot compare different sources, splits or history IDs')
summary=[{'report':path,**{k:r.get(k) for k in ['n','exact_match','macro_f1','invalid_rate','ner_exact_spans','record_errors']}} for path,r in zip(a.reports,reports)]
dump(a.output,summary)
print(__import__('json').dumps(summary,indent=2,ensure_ascii=False))

"""Timing must cover the run, survive failures, and never overwrite an old run."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from common.data import dump, load
from common.runtime import new_output, timed_run, timed_phase


def test_timing_preserves_metrics_and_measures_all_phases(tmp_path):
    @timed_run('evaluate')
    def run():
        out=new_output(tmp_path/'run')
        assert load(out/'timing.json')['status']=='running'
        with timed_phase('model_load'):
            pass
        with timed_phase('inference', samples=3):
            pass
        dump(out/'metrics.json', {'macro_f1':.7,'metadata':{'seconds':123}})
        return 'unchanged result'
    assert run()=='unchanged result'
    timing=load(tmp_path/'run/timing.json')
    assert timing['total_seconds']>=sum(p['seconds'] for p in timing['phases'])>=0
    assert timing['started_at'] and timing['finished_at']
    metrics=load(tmp_path/'run/metrics.json')
    assert metrics['macro_f1']==.7 and metrics['metadata']['seconds']==123
    assert metrics['metadata']['timing']==timing


def test_failed_run_saves_time_without_masking_exception(tmp_path):
    @timed_run('train')
    def run():
        out=new_output(tmp_path/'failed')
        (out/'metrics.json').write_text('partial export')
        with timed_phase('fit_and_validation'):
            raise ValueError('original failure')
    with pytest.raises(ValueError,match='original failure'):
        run()
    timing=load(tmp_path/'failed/timing.json')
    assert timing['status']=='failed' and timing['total_seconds']>=0
    assert timing['phases'][0]['status']=='failed'
    # Context is reset: later unrelated operations must not alter the failed run.
    before=(tmp_path/'failed/timing.json').read_bytes()
    new_output(tmp_path/'unrelated')
    assert not (tmp_path/'unrelated/timing.json').exists()
    assert (tmp_path/'failed/timing.json').read_bytes()==before


def test_rejected_output_does_not_overwrite_existing_timing(tmp_path):
    old=tmp_path/'existing';old.mkdir()
    dump(old/'timing.json', {'old':True})
    @timed_run('train')
    def run():
        new_output(old)
    with pytest.raises(ValueError,match='Output already has files'):
        run()
    assert load(old/'timing.json')=={'old':True}

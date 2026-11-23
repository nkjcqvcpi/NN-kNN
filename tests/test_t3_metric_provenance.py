"""Public metric provenance: the metric is bound through its t3_need_matching run unit only."""
import pytest
import torch

from model.common import tracking as tr
from model.t3.metric_provenance import metric_provenance


@pytest.fixture
def runs(tmp_path,monkeypatch):
    path=tmp_path/'runs';path.mkdir()
    monkeypatch.setenv('NNKNN_RUNS',str(path))
    for name in ('NNKNN_RESUME','NNKNN_RESUME_EVERY_SECONDS','NNKNN_RESUME_EVERY_STEPS'):monkeypatch.delenv(name,raising=False)
    return path


def recorded(runs,*,actual=False,selected_epoch=0,protocol_edit=None,seed=9):
    """A complete unit runs/t3/p/s<seed>/ shaped like the one t3_need_matching records."""
    binding={'sha256':'need'} if actual else None
    protocol=dict(public_dev_annotations_used=False,selection='tune_recall',epochs=20,dimensions=256,seeds=[seed],
        need_objective='hard_negative',term_frequency='count',fit_ids=['fit'],tune_ids=['tune'],data_source={'sha256':'data'},
        source_fingerprint={'sha256':'source'},query_binding=binding,
        query_source='actual frozen host request' if actual else 'original public question')
    protocol.update(protocol_edit or {})
    config=dict(tool='t3_need_matching',seed=seed,epochs=20,dimensions=256,objective='hard_negative',selection='tune_recall',
        term_frequency='count',data_sha256='data',samples_sha256='samples',need_queries_sha256='need' if actual else None)
    key=tr.RunKey('t3','p',(f's{seed}',))
    with tr.open_run(key,config=config) as run:
        run.log_record('protocol',protocol)
        run.log_final(dict(seed=seed,selected_epoch=selected_epoch))
        run.save_state('metric',dict(selected_metric=dict(feature_weights=torch.ones(1,256))),step=20)
    return tr.unit_dir(key)


@pytest.mark.parametrize('actual',[False,True])
def test_records_real_selection_and_epoch_zero_without_prespecified_seed_claim(runs,actual):
    result=metric_provenance(recorded(runs,actual=actual))
    assert result['seed']==9 and result['selected_epoch']==0
    assert result['selection']=='tune_recall' and result['need_objective']=='hard_negative'
    assert result['query_source']==('actual frozen host request' if actual else 'original public question')
    assert 'does not establish' in result['boundary']


def test_rejects_replaced_metric_bytes(runs):
    unit=recorded(runs);weights=unit/'weights'/'metric.pt'
    blob=bytearray(weights.read_bytes());blob[-1]^=1;weights.write_bytes(bytes(blob))
    with pytest.raises(ValueError,match='not complete and verified'):metric_provenance(unit)


def test_rejects_a_selected_epoch_outside_the_budget(runs):
    with pytest.raises(ValueError,match='selected epoch'):metric_provenance(recorded(runs,selected_epoch=21))


def test_rejects_unbound_actual_need_claim(runs):
    unit=recorded(runs,protocol_edit=dict(query_source='actual frozen host request'))
    with pytest.raises(ValueError,match='query source'):metric_provenance(unit)


def test_rejects_a_legacy_metric_file(tmp_path):
    legacy=tmp_path/'s9'/'metric.pt';legacy.parent.mkdir();legacy.write_bytes(b'recorded selected state')
    with pytest.raises(ValueError,match='not a t3_need_matching run unit'):metric_provenance(legacy)

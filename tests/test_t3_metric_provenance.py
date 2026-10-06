import hashlib
import json

import pytest

from model.t3.metric_provenance import metric_provenance


def recorded(tmp_path,*,actual=False):
    folder=tmp_path/'s9';folder.mkdir()
    checkpoint=folder/'metric.pt';checkpoint.write_bytes(b'recorded selected state')
    protocol=dict(public_dev_annotations_used=False,selection='tune_recall',epochs=20,
        need_objective='hard_negative',fit_ids=['fit'],tune_ids=['tune'],data_source={'sha256':'data'},
        source_fingerprint={'sha256':'source'})
    if actual:
        protocol.update(query_source='actual frozen host request',query_binding={'sha256':'need'})
    run=dict(seed=9,selected_epoch=0,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'summary.json').write_text(json.dumps(dict(protocol=protocol,runs=[run])))
    (folder/'summary.json').write_text(json.dumps(dict(seed=9,selected_epoch=0)))
    return checkpoint


@pytest.mark.parametrize('actual',[False,True])
def test_records_real_selection_and_epoch_zero_without_prespecified_seed_claim(tmp_path,actual):
    result=metric_provenance(recorded(tmp_path,actual=actual))
    assert result['seed']==9 and result['selected_epoch']==0
    assert result['selection']=='tune_recall' and result['need_objective']=='hard_negative'
    assert result['query_source']==('actual frozen host request' if actual else 'original public question')
    assert 'does not establish' in result['boundary']


def test_rejects_replaced_metric_bytes(tmp_path):
    checkpoint=recorded(tmp_path);checkpoint.write_bytes(b'other metric')
    with pytest.raises(ValueError,match='uniquely bound'):metric_provenance(checkpoint)


def test_rejects_edited_seed_selection(tmp_path):
    checkpoint=recorded(tmp_path)
    (checkpoint.parent/'summary.json').write_text(json.dumps(dict(seed=9,selected_epoch=3)))
    with pytest.raises(ValueError,match='seed summary'):metric_provenance(checkpoint)


def test_rejects_unbound_actual_need_claim(tmp_path):
    checkpoint=recorded(tmp_path)
    protocol=json.loads((tmp_path/'protocol.json').read_text());protocol['query_source']='actual frozen host request'
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    summary=json.loads((tmp_path/'summary.json').read_text());summary['protocol']=protocol
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='query source'):metric_provenance(checkpoint)

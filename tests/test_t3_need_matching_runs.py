"""t3_need_matching: run-unit recording (--study) computes what the file-based path computes."""
import hashlib
import json
import random

import pytest
import torch

from model.common import tracking as tr
from model.t3.metric_provenance import load_metric_checkpoint,metric_checkpoint_sha256,metric_provenance
import tools.t3_need_matching as nm

WORDS=[f'w{i}' for i in range(60)]


@pytest.fixture
def data(tmp_path):
    """A small SQuAD1.1 mirror (72 passages,8 articles),its manifest and a fit/tune sample file."""
    stream=random.Random(5);articles=[];fit=[];tune=[]
    for a in range(8):
        paragraphs=[]
        for p in range(9):
            context=' '.join(stream.choice(WORDS) for _ in range(12))
            qid=f'q{a}_{p}';answer=context.split()[0]
            question='which '+' '.join(stream.sample(context.split(),3))
            paragraphs.append(dict(context=context,qas=[dict(id=qid,question=question,answers=[dict(text=answer,answer_start=0)])]))
            if p<2:(fit if a<5 else tune).append(qid)
        articles.append(dict(title=f'article{a}',paragraphs=paragraphs))
    root=tmp_path/'data';root.mkdir()
    raw=json.dumps(dict(version='1.1',data=articles)).encode();(root/'squad_train_v11.json').write_bytes(raw)
    (root/'manifest_mirror_complete.json').write_text(json.dumps(dict(files=[dict(name='squad_train_v11.json',
        sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))])))
    samples=tmp_path/'samples.json'
    samples.write_text(json.dumps(dict(samples={'squad_train_v11.json':dict(fit_ids=fit,tune_ids=tune)})))
    return root,samples


@pytest.fixture
def runs(tmp_path,monkeypatch):
    path=tmp_path/'runs';path.mkdir()
    monkeypatch.setenv('NNKNN_RUNS',str(path))
    monkeypatch.delenv('NNKNN_RESUME',raising=False)
    monkeypatch.delenv('NNKNN_RESUME_EVERY_SECONDS',raising=False)
    monkeypatch.delenv('NNKNN_RESUME_EVERY_STEPS',raising=False)
    return path


def run(data,output,*extra):
    root,samples=data
    nm.main(['--data-root',str(root),'--samples',str(samples),'--output',str(output),'--seeds','3','4',
             '--epochs','3','--dimensions','32',*extra])


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def untimed(result):
    return {k:v for k,v in result.items() if k not in ('training_seconds','checkpoint_sha256')}


def test_study_records_the_same_numbers_and_writes_no_per_seed_files(data,runs,tmp_path):
    run(data,tmp_path/'files')
    run(data,tmp_path/'study','--study','tm')
    files,study=tmp_path/'files',tmp_path/'study'
    assert sorted(p.name for p in study.iterdir())==['protocol.json','summary.json']
    legacy=json.loads((files/'summary.json').read_text());recorded=json.loads((study/'summary.json').read_text())
    protocol=json.loads((study/'protocol.json').read_text())
    assert protocol.pop('study')=='tm' and protocol==json.loads((files/'protocol.json').read_text())
    assert recorded['protocol']==dict(protocol,study='tm')
    for seed,old,new in zip((3,4),legacy['runs'],recorded['runs']):
        unit=runs/'t3'/'tm'/f's{seed}'
        info=tr.run_status(unit);assert info.status=='complete' and not tr.verify_run(unit)
        data_=tr.load_run(unit)
        assert untimed(old)==untimed(new)==untimed(json.loads((files/f's{seed}'/'summary.json').read_text()))
        assert untimed(nm.seed_result(data_))==untimed(old)
        assert new['checkpoint_sha256']==data_.states()['metric']['file']['sha256']
        assert equal(tr.load_state(unit,'metric'),torch.load(files/f's{seed}'/'metric.pt',weights_only=True))
        assert data_.records('training_events')==json.loads((files/f's{seed}'/'training_events.json').read_text())
        assert data_.record('protocol')==recorded['protocol']
        assert [s for s,_ in data_.series('tune/mean_need_loss')]==[0,1,2,3]
        assert [s for s,_ in data_.series('train/mean_loss')]==[1,2,3]
        assert data_.config==nm.unit_config(type('A',(),dict(epochs=3,dimensions=32,objective='softmax',selection='tune_loss',
            term_frequency='count'))(),seed,source=protocol['data_source'],
            samples_sha256=hashlib.sha256(data[1].read_bytes()).hexdigest(),query_binding=None)
        assert sorted(p.name for p in (unit/'weights').iterdir())==['metric.pt']


def test_rerun_skips_complete_units_and_writes_identical_summary_bytes(data,runs,tmp_path,monkeypatch):
    run(data,tmp_path/'first','--study','tm')
    def no_training(*args,**kwargs):raise AssertionError('a complete unit was trained again')
    monkeypatch.setattr(nm,'need_loss',no_training)
    run(data,tmp_path/'again','--study','tm')
    for name in ('summary.json','protocol.json'):
        assert (tmp_path/'first'/name).read_bytes()==(tmp_path/'again'/name).read_bytes()
    assert all(tr.run_status(runs/'t3'/'tm'/f's{seed}').sessions==1 for seed in (3,4))


def test_a_changed_setting_does_not_reuse_a_complete_unit(data,runs,tmp_path):
    run(data,tmp_path/'first','--study','tm')
    with pytest.raises(tr.ConfigMismatchError):run(data,tmp_path/'longer','--study','tm','--epochs','4')


def test_an_interrupted_seed_resumes_to_the_uninterrupted_result(data,runs,tmp_path,monkeypatch):
    run(data,tmp_path/'whole','--study','whole')
    monkeypatch.setenv('NNKNN_RESUME_EVERY_STEPS','1')
    real,calls=nm.need_loss,[0]
    def crash(*args,**kwargs):
        calls[0]+=1
        if calls[0]==16+26+26+4:raise RuntimeError('killed in epoch 3')  # baseline (10 fit+6 tune),epochs 1-2,then training
        return real(*args,**kwargs)
    monkeypatch.setattr(nm,'need_loss',crash)
    with pytest.raises(RuntimeError,match='killed'):run(data,tmp_path/'crashed','--study','cut')
    assert tr.run_status(runs/'t3'/'cut'/'s3').status=='failed'
    monkeypatch.setattr(nm,'need_loss',real)
    run(data,tmp_path/'resumed','--study','cut')
    whole,resumed=(json.loads((tmp_path/name/'summary.json').read_text()) for name in ('whole','resumed'))
    assert [untimed(r) for r in whole['runs']]==[untimed(r) for r in resumed['runs']]
    info=tr.run_status(runs/'t3'/'cut'/'s3');assert info.status=='complete' and info.resumed and info.sessions==2
    a,b=(tr.load_run(runs/'t3'/name/'s3') for name in ('whole','cut'))
    assert a.records('training_events')==b.records('training_events')
    assert a.states()['metric']['state_sha256']==b.states()['metric']['state_sha256']
    assert equal(tr.load_state(a.path,'metric'),tr.load_state(b.path,'metric'))
    assert a.series('train/mean_loss')==b.series('train/mean_loss')


def test_public_pilots_read_the_metric_from_a_unit_like_from_metric_pt(data,runs,tmp_path):
    run(data,tmp_path/'files')
    run(data,tmp_path/'study','--study','tm')
    legacy,unit=tmp_path/'files'/'s3'/'metric.pt',runs/'t3'/'tm'/'s3'
    old,new=metric_provenance(legacy),metric_provenance(unit)
    assert old.keys()==new.keys()
    for key in old:
        if key not in ('checkpoint_sha256','artifacts'):assert old[key]==new[key],key
    assert new['checkpoint_sha256']==metric_checkpoint_sha256(unit)==hashlib.sha256((unit/'weights'/'metric.pt').read_bytes()).hexdigest()
    assert metric_checkpoint_sha256(legacy)==old['checkpoint_sha256']==hashlib.sha256(legacy.read_bytes()).hexdigest()
    assert new['artifacts']['run_unit']['run_id']=='t3/tm/s3'
    assert equal(load_metric_checkpoint(unit),load_metric_checkpoint(legacy))
    weights=unit/'weights'/'metric.pt';blob=bytearray(weights.read_bytes());blob[-1]^=1;weights.write_bytes(bytes(blob))
    with pytest.raises(ValueError,match='not complete and verified'):metric_provenance(unit)


@pytest.mark.parametrize('field,value',[('selection','tune_recall'),('need_objective','hard_negative'),('term_frequency','binary'),
                                        ('dimensions',64),('epochs',4),('query_binding',dict(sha256='0'*64))])
def test_unit_provenance_rejects_a_protocol_record_that_differs_from_the_unit_config(data,runs,tmp_path,monkeypatch,field,value):
    run(data,tmp_path/'study','--study','tm')
    unit=runs/'t3'/'tm'/'s3';real=tr.load_run
    class Changed:
        def __init__(self,data):self._data=data
        def __getattr__(self,name):return getattr(self._data,name)
        def record(self,name):
            value_=self._data.record(name)
            return {**value_,field:value} if name=='protocol' else value_
    monkeypatch.setattr(tr,'load_run',lambda *a,**k:Changed(real(*a,**k)))
    with pytest.raises(ValueError,match='protocol differs from its config'):metric_provenance(unit)

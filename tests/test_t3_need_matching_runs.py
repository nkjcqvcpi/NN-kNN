"""t3_need_matching: recording is mandatory (--study),every seed is a run unit,a complete unit is never trained again."""
import hashlib
import json
from pathlib import Path
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


def run(data,output,*extra,seeds=('3','4')):
    root,samples=data
    nm.main(['--data-root',str(root),'--samples',str(samples),*(['--output',str(output)] if output else []),
             '--seeds',*seeds,'--epochs','3','--dimensions','32',*extra])


SEED_CALLS=16+3*(10+16)+16  # need_loss calls of one seed: baseline fit+tune,3 epochs (10 steps+evaluation),selected evaluation


class Counter:
    """Wraps need_loss to count its calls (training steps and evaluations both go through it)."""
    def __init__(self,real):self.real,self.calls=real,0
    def __call__(self,*args,**kwargs):
        self.calls+=1
        return self.real(*args,**kwargs)


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def untimed(result):
    return {k:v for k,v in result.items() if k not in ('training_seconds','checkpoint_sha256')}


def test_study_is_required(data,runs,tmp_path):
    with pytest.raises(SystemExit) as exc:run(data,tmp_path/'out')
    assert exc.value.code==2 and not (tmp_path/'out').exists() and not any(runs.iterdir())


def test_a_study_records_every_seed_as_a_unit_and_writes_only_the_two_summaries(data,runs,tmp_path):
    run(data,tmp_path/'study','--study','tm')
    study=tmp_path/'study'
    assert sorted(p.name for p in study.iterdir())==['protocol.json','summary.json']
    recorded=json.loads((study/'summary.json').read_text());protocol=json.loads((study/'protocol.json').read_text())
    assert protocol['study']=='tm' and recorded['protocol']==protocol and [r['seed'] for r in recorded['runs']]==[3,4]
    for seed,new in zip((3,4),recorded['runs']):
        unit=runs/'t3'/'tm'/f's{seed}'
        info=tr.run_status(unit);assert info.status=='complete' and not tr.verify_run(unit)
        data_=tr.load_run(unit)
        assert nm.seed_result(data_)=={k:v for k,v in new.items() if k!='checkpoint_sha256'}
        assert new['checkpoint_sha256']==data_.states()['metric']['file']['sha256']
        state=tr.load_state(unit,'metric')
        assert sorted(state)==['final_metric','final_optimizer','initial_metric','selected_metric','selected_optimizer']
        assert state['selected_metric']['feature_weights'].shape==(1,32)
        assert len(data_.records('training_events'))==new['actual_training_steps']==3*10
        assert data_.record('protocol')==protocol
        assert [s for s,_ in data_.series('tune/mean_need_loss')]==[0,1,2,3]
        assert [s for s,_ in data_.series('train/mean_loss')]==[1,2,3]
        assert data_.config==nm.unit_config(type('A',(),dict(epochs=3,dimensions=32,objective='softmax',selection='tune_loss',
            term_frequency='count'))(),seed,source=protocol['data_source'],
            samples_sha256=hashlib.sha256(data[1].read_bytes()).hexdigest(),query_binding=None)
        assert sorted(p.name for p in (unit/'weights').iterdir())==['metric.pt']


def test_the_default_output_is_under_results(data,runs,tmp_path,monkeypatch):
    assert nm.RESULTS==Path(nm.__file__).resolve().parents[1]/'results'/'t3'/'s1_single_hop_retrieval'/'need_matching'
    monkeypatch.setattr(nm,'RESULTS',tmp_path/'results')
    run(data,None,'--study','tm')
    assert sorted(p.name for p in (tmp_path/'results'/'tm').iterdir())==['protocol.json','summary.json']
    assert nm.default_output('x')==nm.RESULTS/'x'


def test_an_output_folder_with_other_files_or_another_study_is_refused(data,runs,tmp_path):
    old=tmp_path/'old';(old/'s3').mkdir(parents=True);(old/'s3'/'metric.pt').write_bytes(b'legacy')
    with pytest.raises(SystemExit):run(data,old,'--study','tm')
    assert (old/'s3'/'metric.pt').read_bytes()==b'legacy' and sorted(p.name for p in old.iterdir())==['s3']
    run(data,tmp_path/'a','--study','a')
    with pytest.raises(SystemExit):run(data,tmp_path/'a','--study','b')


def test_rerun_skips_complete_units_and_writes_identical_summary_bytes(data,runs,tmp_path,monkeypatch):
    run(data,tmp_path/'first','--study','tm')
    before={name:(tmp_path/'first'/name).read_bytes() for name in ('summary.json','protocol.json')}
    def no_training(*args,**kwargs):raise AssertionError('a complete unit was trained again')
    monkeypatch.setattr(nm,'need_loss',no_training)
    run(data,tmp_path/'first','--study','tm')  # same output folder: its two summaries are rewritten
    run(data,tmp_path/'again','--study','tm')
    for name,raw in before.items():
        assert (tmp_path/'first'/name).read_bytes()==raw==(tmp_path/'again'/name).read_bytes()
    assert all(tr.run_status(runs/'t3'/'tm'/f's{seed}').sessions==1 for seed in (3,4))


def test_a_changed_setting_does_not_reuse_a_complete_unit(data,runs,tmp_path):
    run(data,tmp_path/'first','--study','tm')
    with pytest.raises(tr.ConfigMismatchError):run(data,tmp_path/'longer','--study','tm','--epochs','4')


def test_a_partly_complete_study_trains_only_the_missing_seed_and_equals_an_uninterrupted_one(data,runs,tmp_path,monkeypatch):
    run(data,tmp_path/'whole','--study','whole')
    counter=Counter(nm.need_loss)
    def crash(*args,**kwargs):
        if counter.calls==SEED_CALLS+5:raise RuntimeError('killed in the baseline of seed 4')
        return counter(*args,**kwargs)
    monkeypatch.setattr(nm,'need_loss',crash)
    with pytest.raises(RuntimeError,match='killed'):run(data,tmp_path/'part','--study','part')
    assert tr.run_status(runs/'t3'/'part'/'s3').status=='complete'
    assert tr.run_status(runs/'t3'/'part'/'s4').status=='failed'
    counter=Counter(counter.real);monkeypatch.setattr(nm,'need_loss',counter)
    run(data,tmp_path/'part','--study','part')
    assert counter.calls==SEED_CALLS  # only seed 4 ran; the complete seed 3 was neither trained nor evaluated again
    assert tr.run_status(runs/'t3'/'part'/'s3').sessions==1 and tr.run_status(runs/'t3'/'part'/'s4').sessions==2
    whole,part=(json.loads((tmp_path/name/'summary.json').read_text()) for name in ('whole','part'))
    assert [untimed(r) for r in whole['runs']]==[untimed(r) for r in part['runs']]
    for seed in (3,4):
        a,b=(tr.load_run(runs/'t3'/name/f's{seed}') for name in ('whole','part'))
        assert a.states()['metric']['state_sha256']==b.states()['metric']['state_sha256']
        assert equal(tr.load_state(a.path,'metric'),tr.load_state(b.path,'metric'))
        assert a.records('training_events')==b.records('training_events')
        assert a.series('tune/mean_need_loss')==b.series('tune/mean_need_loss')


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
    assert list((runs/'t3'/'cut'/'s3'/'weights').glob('resume/*'))  # resume files stay after completion (O5)


def test_public_pilots_read_the_metric_only_from_a_unit(data,runs,tmp_path):
    run(data,tmp_path/'study','--study','tm')
    unit=runs/'t3'/'tm'/'s3'
    binding=metric_provenance(unit)
    assert binding['seed']==3 and binding['selection']=='tune_loss' and binding['artifacts']['run_unit']['run_id']=='t3/tm/s3'
    assert binding['checkpoint_sha256']==metric_checkpoint_sha256(unit)==hashlib.sha256((unit/'weights'/'metric.pt').read_bytes()).hexdigest()
    assert equal(load_metric_checkpoint(unit),tr.load_state(unit,'metric'))
    legacy=tmp_path/'old'/'s3'/'metric.pt';legacy.parent.mkdir(parents=True)
    torch.save(tr.load_state(unit,'metric'),legacy)
    for reader in (metric_provenance,metric_checkpoint_sha256,load_metric_checkpoint):
        with pytest.raises(ValueError,match='not a t3_need_matching run unit'):reader(legacy)
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

import hashlib
import json

import pytest

from model.t3.benchmarks import Gold,answer_metrics,hotpot_metrics,load_hotpot,load_squad,select_by_id


def source(tmp_path,obj):
    path=tmp_path/'data.json';raw=json.dumps(obj).encode();path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


def test_hotpot_annotations_separate_and_originals_preserved(tmp_path):
    obj=[dict(_id='q1',question='Which city?',answer='City A',context=[['Title',[' sentence A.','Sentence B.']]],
              supporting_facts=[['Title',1]])]
    path,digest=source(tmp_path,obj)
    tasks,gold=load_hotpot(path,expected_sha256=digest)
    assert tasks[0].host_task()==dict(question_id='q1',question='Which city?')
    assert json.loads(tasks[0].cases[0].content)==dict(title='Title',sentences=obj[0]['context'][0][1])
    assert 'supporting_facts' not in tasks[0].cases[0].content and 'City A' not in tasks[0].cases[0].content
    assert gold['q1'].supporting_facts==(('Title',1),)
    assert load_hotpot(path,expected_sha256=digest)[0]==tasks
    with pytest.raises(ValueError,match='hash mismatch'):load_hotpot(path,expected_sha256='0'*64)


def test_squad_shared_passage_and_span_validation(tmp_path):
    obj=dict(version='1.1',data=[dict(title='T',paragraphs=[dict(context='Blue city.',qas=[
        dict(id='q1',question='Color?',answers=[dict(text='Blue',answer_start=0)]),
        dict(id='q2',question='Place?',answers=[dict(text='city',answer_start=5)])])])])
    path,digest=source(tmp_path,obj)
    tasks,gold=load_squad(path,expected_sha256=digest)
    assert tasks[0].cases==tasks[1].cases and tasks[0].group==tasks[1].group=='T'
    assert gold['q1'].answers==('Blue',)
    assert select_by_id(tasks,count=1,seed=42)==select_by_id(list(reversed(tasks)),count=1,seed=42)
    obj['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']=1
    path,digest=source(tmp_path,obj)
    with pytest.raises(ValueError,match='align'):load_squad(path,expected_sha256=digest)


def test_metrics_alias_multiset_special_answer_and_joint_support():
    score=answer_metrics('The red red car!',Gold(('red car','red red car')))
    assert score['em']==score['f1']==1
    assert answer_metrics('red red',Gold(('red car',)))['f1']==pytest.approx(.5)
    assert answer_metrics('yes please',Gold(('yes',)),hotpot=True)['f1']==0
    assert answer_metrics('',Gold(('the',)))['f1']==1
    gold=Gold(('red car',),(('T',0),('U',1)))
    score=hotpot_metrics('red',[['T',0],['T',0]],gold)
    assert score['f1']==pytest.approx(2/3) and score['sp_f1']==pytest.approx(2/3)
    assert score['joint_precision']==1 and score['joint_recall']==.25
    assert score['joint_f1']==pytest.approx(.4)
    with pytest.raises(ValueError):hotpot_metrics('red',[['T',True]],gold)


def test_invalid_support_is_rejected_instead_of_leaking_or_dropping(tmp_path):
    path,digest=source(tmp_path,[dict(_id='q',question='Q?',answer='A',context=[['T',['S']]],supporting_facts=[['T',1]])])
    with pytest.raises(ValueError,match='align'):load_hotpot(path,expected_sha256=digest)
    tasks,gold=load_hotpot(path,expected_sha256=digest,allow_unaligned_support=True)
    assert len(tasks)==1 and gold['q'].supporting_facts==gold['q'].unavailable_supporting_facts==(('T',1),)
    assert tasks[0].host_task()==dict(question_id='q',question='Q?')

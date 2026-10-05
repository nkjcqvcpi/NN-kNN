import copy
import json
from types import SimpleNamespace
import pytest
from model.t3.host_schema import request_prompt
from model.t3.need_queries import load_need_queries


def fixture():
    task=SimpleNamespace(question_id='fit',question='Original public question?')
    request=dict(ready=False,need='Original public question? Please provide evidence',requested_types=['evidence'],observable_task_state='awaiting evidence')
    call=dict(stage='request',source_prompt=request_prompt(task.question),prompt='<user>'+request_prompt(task.question)+'</user>',output=json.dumps(request),input_tokens=40,output_tokens=50)
    artifact=dict(protocol=dict(version=1,dataset='squad_train_v11.json',data_sha256='a'*64,sample_sha256='b'*64,query_source='actual frozen host request',public_dev_annotations_used=False,
        host=dict(revision='f'*40,files=[dict(name='model.safetensors',sha256='d'*64,bytes=128)])),
        records=[dict(question_id='fit',question=task.question,request=request,call=call,error=None)])
    return task,artifact


def load(tmp_path,task,artifact):
    path=tmp_path/'needs.json';path.write_text(json.dumps(artifact),encoding='utf-8')
    return load_need_queries(path,[task],data_sha256='a'*64,sample_sha256='b'*64)


def test_actual_emitted_need_is_kept_without_copying_original_question(tmp_path):
    task,artifact=fixture();before=copy.deepcopy(artifact)
    queries,binding=load(tmp_path,task,artifact)
    assert queries['fit']==artifact['records'][0]['request']['need'] and queries['fit']!=task.question
    assert artifact==before and binding['records']==1 and len(binding['sha256'])==64


def test_orphan_missing_or_failed_queries_cannot_be_filtered_or_filled(tmp_path):
    task,artifact=fixture();artifact['records'][0]['error']='truncated';artifact['records'][0]['request']=None
    with pytest.raises(ValueError,match='unobserved'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['records']=[]
    with pytest.raises(ValueError,match='complete'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['records'][0]['question_id']='dev'
    with pytest.raises(ValueError,match='identity'):load(tmp_path,task,artifact)


def test_changed_needs_prompt_or_data_binding_are_rejected(tmp_path):
    task,artifact=fixture();artifact['records'][0]['request']['need']=task.question
    with pytest.raises(ValueError,match='emitted'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['records'][0]['call']['source_prompt']+=' supplied answer'
    with pytest.raises(ValueError,match='prompt'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['protocol']['data_sha256']='c'*64
    with pytest.raises(ValueError,match='binding'):load(tmp_path,task,artifact)


def test_dev_annotations_duplicate_records_and_wrong_stages_are_rejected(tmp_path):
    task,artifact=fixture();artifact['protocol']['public_dev_annotations_used']=True
    with pytest.raises(ValueError,match='train-only'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['records']*=2
    with pytest.raises(ValueError,match='identity'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['records'][0]['call']['stage']='answer'
    with pytest.raises(ValueError,match='stage'):load(tmp_path,task,artifact)


def test_unbound_host_or_duplicate_host_files_are_rejected(tmp_path):
    task,artifact=fixture();artifact['protocol'].pop('host')
    with pytest.raises(ValueError,match='host revision'):load(tmp_path,task,artifact)
    _,artifact=fixture();artifact['protocol']['host']['files']*=2
    with pytest.raises(ValueError,match='host file'):load(tmp_path,task,artifact)

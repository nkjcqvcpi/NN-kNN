import pytest
from model.t3.host_schema import decision_schema,decoder_schema,validate_decision


def test_request_schema_never_fills_missing_fields_or_changes_need():
    request=dict(ready=False,need='actual generated need',requested_types=['evidence'],observable_task_state='public state')
    assert validate_decision(request,'request') is request
    missing=dict(request);missing.pop('requested_types')
    with pytest.raises(ValueError):validate_decision(missing,'request')
    assert 'requested_types' not in missing
    with pytest.raises(ValueError):validate_decision(dict(request,ready=True),'request')
    assert decision_schema('request')['properties']['ready']['const'] is False


def test_ordered_citation_checks_are_stricter_than_decoder_item_union():
    answer=dict(ready=True,answer='public answer',supporting_facts=[['Title',0]])
    assert validate_decision(answer,'answer') is answer
    for pair in ([0,'Title'],['Title',True],['Title',-1],['Title','0']):
        with pytest.raises(ValueError):validate_decision(dict(answer,supporting_facts=[pair]),'answer')
    assert 'anyOf' in decision_schema('answer')['properties']['supporting_facts']['items']['items']


def test_continuation_is_explicit_request_or_answer():
    assert len(decision_schema('continuation')['anyOf'])==2
    validate_decision(dict(ready=True,answer='UNKNOWN',supporting_facts=[]),'continuation')
    with pytest.raises(ValueError):validate_decision(dict(ready=False,answer='invented'),'continuation')


def test_backend_schema_exposes_weaker_boolean_contract_without_mutating_it():
    declared=decision_schema('request')
    supported=decoder_schema('request')
    assert declared['properties']['ready']==dict(const=False)
    assert supported['properties']['ready']==dict(type='boolean')
    assert supported['required']==declared['required']
    assert decision_schema('request')==declared

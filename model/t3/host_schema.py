"""Explicit public decision contracts for optional constrained decoding.

Formatting constrains keys/types,not answer truth or relevance. Runtime
validation checks ordered citation pairs more strictly than the decoder's item
union. No missing need,answer,type or state is filled by the controller.
"""
import copy
from .retrieval import TYPES,Request


def decision_schema(stage,*,allowed_types=('evidence',)):
    if stage not in {'request','answer','continuation'}:
        raise ValueError('unsupported host stage')
    if not isinstance(allowed_types,(tuple,list)) or not allowed_types or any(not isinstance(t,str) or t not in TYPES for t in allowed_types) or len(set(allowed_types))!=len(allowed_types):
        raise ValueError('explicit supported artifact types required')
    request=dict(type='object',additionalProperties=False,required=['ready','need','requested_types','observable_task_state'],properties=dict(
        ready=dict(const=False),need=dict(type='string',minLength=1,maxLength=4096),
        requested_types=dict(type='array',minItems=1,maxItems=len(allowed_types),items=dict(type='string',enum=list(allowed_types))),
        observable_task_state=dict(type='string',maxLength=8192)))
    # The optional decoder supports an item union,not heterogeneous positional
    # tuples. validate_decision enforces [title,index] after generation.
    citation=dict(type='array',minItems=2,maxItems=2,items=dict(anyOf=[dict(type='string',minLength=1,maxLength=256),dict(type='integer',minimum=0)]))
    answer=dict(type='object',additionalProperties=False,required=['ready','answer','supporting_facts'],properties=dict(
        ready=dict(const=True),answer=dict(type='string',minLength=1,maxLength=512),
        supporting_facts=dict(type='array',maxItems=8,items=citation)))
    return request if stage=='request' else answer if stage=='answer' else dict(anyOf=[request,answer])


def validate_decision(decision,stage,*,allowed_types=('evidence',)):
    decision_schema(stage,allowed_types=allowed_types)
    if not isinstance(decision,dict) or type(decision.get('ready')) is not bool:
        raise ValueError('host requires explicit boolean readiness')
    actual='answer' if decision['ready'] else 'request'
    if stage!='continuation' and actual!=stage:
        raise ValueError('host readiness must match declared stage')
    if actual=='request':
        if set(decision)!={'ready','need','requested_types','observable_task_state'}:
            raise ValueError('host request must supply exact public fields')
        request=Request(decision['need'],decision['requested_types'],observable_task_state=decision['observable_task_state'])
        if not isinstance(decision['requested_types'],list) or not 1<=len(decision['requested_types'])<=len(allowed_types) or not set(request.requested_types)<=set(allowed_types):
            raise ValueError('host request types exceed declared routes')
    else:
        if set(decision)!={'ready','answer','supporting_facts'}:
            raise ValueError('host answer must supply exact public fields')
        if not isinstance(decision['answer'],str) or not decision['answer'].strip() or len(decision['answer'])>512:
            raise ValueError('host answer must be a bounded nonempty public string')
        support=decision['supporting_facts']
        if not isinstance(support,list) or len(support)>8 or any(not isinstance(p,list) or len(p)!=2 or
                not isinstance(p[0],str) or not p[0].strip() or len(p[0])>256 or type(p[1]) is not int or p[1]<0 for p in support):
            raise ValueError('host citations require bounded ordered title/index pairs')
    return decision


def decoder_schema(stage,*,allowed_types=('evidence',)):
    """Backend0.11.3 cannot parse boolean const; enforce readiness afterwards.

    It also does not fully enforce numeric minima or positional tuple types.
    Runtime validation stays authoritative; grammar is not a truth validator.
    """
    schema=copy.deepcopy(decision_schema(stage,allowed_types=allowed_types))
    def convert(value):
        if isinstance(value,dict):
            if type(value.get('const')) is bool:
                value.pop('const');value['type']='boolean'
            for child in value.values():convert(child)
        elif isinstance(value,list):
            for child in value:convert(child)
    convert(schema)
    return schema

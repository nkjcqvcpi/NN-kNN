"""Source-checked public QA adapters; annotations never enter host payloads.

Hotpot uses its provided distractor context, not the full Wikipedia corpus.
SQuAD provides reading passages; a retrieval experiment must declare its corpus.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import string

from .retrieval import Case


@dataclass(frozen=True)
class PublicTask:
    question_id: str
    question: str
    cases: tuple[Case, ...]
    group: str

    def __post_init__(self):
        if not all(isinstance(v,str) and v.strip() for v in (self.question_id,self.question,self.group)):
            raise ValueError('public task requires question, ID and grouping')
        if not self.cases or not isinstance(self.cases,tuple) or any(not isinstance(c,Case) for c in self.cases):
            raise ValueError('public task requires immutable original cases')

    def host_task(self):
        # Passage inventories are supplied through retrieval, not bundled with
        # the question. No answer spans or supporting annotations are exposed.
        return dict(question_id=self.question_id, question=self.question)


@dataclass(frozen=True)
class Gold:
    answers: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...] = ()
    unavailable_supporting_facts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.answers,tuple) or not self.answers or any(not isinstance(a,str) for a in self.answers):
            raise ValueError('gold answers require an immutable nonempty string tuple')


def _read(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('explicit lowercase source SHA256 required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('dataset source hash mismatch')
    return json.loads(raw)


def _case(dataset, source_sha256, key, title, body):
    original = dict(title=title, **body)
    content = json.dumps(original, ensure_ascii=False, separators=(',', ':'))
    source = f'{dataset}:{source_sha256}:{key}'
    case_id = int.from_bytes(hashlib.sha256((source+'\0'+content).encode()).digest()[:8], 'big') & ((1 << 63)-1)
    return Case(case_id, content, 'evidence', source, 'global', validated=True)


def _unique(tasks, gold):
    if len({t.question_id for t in tasks}) != len(tasks) or len(gold) != len(tasks):
        raise ValueError('dataset question IDs must be unique')
    inventory = {}
    for task in tasks:
        for case in task.cases:
            if case.case_id in inventory and case != inventory[case.case_id]:
                raise ValueError('stable case ID collision')
            inventory[case.case_id] = case
    return tasks, gold


def load_squad(path, *, expected_sha256):
    obj = _read(path, expected_sha256)
    if obj.get('version') != '1.1':
        raise ValueError('adapter supports answerable SQuAD1.1 only')
    tasks, gold = [], {}
    for article in obj['data']:
        title = article['title']
        for index, paragraph in enumerate(article['paragraphs']):
            passage = paragraph['context']
            if not isinstance(title, str) or not isinstance(passage, str) or not passage:
                raise ValueError('original title/passage required')
            case = _case('squad1.1', expected_sha256, f'{title}:{index}', title, dict(passage=passage))
            for qa in paragraph['qas']:
                answers = qa['answers']
                if not answers or any(passage[a['answer_start']:a['answer_start']+len(a['text'])] != a['text'] for a in answers):
                    raise ValueError('answer annotations must align original passage')
                qid = qa['id']
                tasks.append(PublicTask(qid,qa['question'],(case,),title))
                gold[qid] = Gold(tuple(a['text'] for a in answers))
    return _unique(tasks, gold)


def load_hotpot(path, *, expected_sha256, allow_unaligned_support=False):
    if type(allow_unaligned_support) is not bool:
        raise ValueError('unaligned annotation handling must be explicit boolean')
    records = _read(path, expected_sha256)
    tasks, gold = [], {}
    for record in records:
        qid = record['_id']
        context = record['context']
        titles = {title: sentences for title, sentences in context}
        if len(titles) != len(context) or any(not isinstance(title,str) or not isinstance(sentences,list)
                or any(not isinstance(s,str) for s in sentences) for title,sentences in context):
            raise ValueError('original unique titles and sentence lists required')
        support = tuple((title,index) for title,index in record['supporting_facts'])
        if any(not isinstance(title,str) or type(index) is not int or index < 0 for title,index in support):
            raise ValueError('support annotations require title and nonnegative index')
        gaps = tuple((title,index) for title,index in support if title not in titles or index >= len(titles[title]))
        if gaps and not allow_unaligned_support:
            raise ValueError('support annotations must align original context')
        cases = tuple(_case('hotpotqa-distractor',expected_sha256,f'{qid}:{title}',title,
                            dict(sentences=sentences)) for title,sentences in context)
        tasks.append(PublicTask(qid,record['question'],cases,qid))
        # Never repair or discard an unavailable gold fact. Explicit opt-in
        # preserves publisher scoring while exposing attainable-evidence limits.
        gold[qid] = Gold((record['answer'],),support,gaps)
    return _unique(tasks, gold)


def select_by_id(tasks, *, count, seed):
    if type(count) is not int or not 0 < count <= len(tasks) or type(seed) is not int:
        raise ValueError('selection requires bounded count and integer seed')
    # Ordering depends on public IDs only, never answers, support, lengths or
    # measured success. Record IDs before host/model experiments.
    return sorted(tasks,key=lambda t:hashlib.sha256(f'{seed}\0{t.question_id}'.encode()).digest())[:count]


def normalize_answer(answer):
    text = ''.join(c for c in answer.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def answer_metrics(prediction, gold, *, hotpot=False):
    if not isinstance(prediction,str) or not gold.answers:
        raise ValueError('public answer string and nonempty gold answers required')
    scores = []
    pred = normalize_answer(prediction)
    for answer in gold.answers:
        target = normalize_answer(answer)
        pt,gt = pred.split(),target.split()
        common = sum((Counter(pt)&Counter(gt)).values())
        special_mismatch = hotpot and pred != target and ({pred,target}&{'yes','no','noanswer'})
        precision = common/len(pt) if common and not special_mismatch else 0.
        recall = common/len(gt) if common and not special_mismatch else 0.
        empty_squad_match = not hotpot and not pt and not gt
        scores.append(dict(em=float(pred==target),f1=1. if empty_squad_match else (2*precision*recall/(precision+recall) if precision+recall else 0.),
                           precision=precision,recall=recall))
    # SQuAD's EM and F1 maximize over aliases independently.
    return {key:max(s[key] for s in scores) for key in scores[0]}


def hotpot_metrics(prediction, supporting_facts, gold):
    answer = answer_metrics(prediction,gold,hotpot=True)
    if any(not isinstance(pair,(list,tuple)) or len(pair)!=2 or not isinstance(pair[0],str)
           or type(pair[1]) is not int or pair[1]<0 for pair in supporting_facts):
        raise ValueError('support predictions require title and nonnegative sentence index')
    predicted,target = set(map(tuple,supporting_facts)),set(gold.supporting_facts)
    matched = len(predicted&target)
    precision = matched/len(predicted) if predicted else 0.
    recall = matched/len(target) if target else 0.
    joint_p,joint_r = precision*answer['precision'],recall*answer['recall']
    return dict(answer,sp_em=float(predicted==target),sp_precision=precision,sp_recall=recall,
                sp_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                joint_em=answer['em']*float(predicted==target),joint_precision=joint_p,joint_recall=joint_r,
                joint_f1=2*joint_p*joint_r/(joint_p+joint_r) if joint_p+joint_r else 0.)

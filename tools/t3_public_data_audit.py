"""Validate source artifacts, annotation separation and public QA scoring.

This emits a prospective engineering sample manifest, not model results.
Reference evaluators must come from the pinned source manifest and are executed
only for scorer comparison. Hotpot dev is never used to train or select models.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from model.common.artifacts import source_fingerprint
from model.t3.benchmarks import answer_metrics,hotpot_metrics,load_hotpot,load_squad,select_by_id


def reference(path):
    spec=importlib.util.spec_from_file_location('reference_'+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    old=sys.modules.get('ujson')
    sys.modules['ujson']=json
    try:spec.loader.exec_module(module)
    finally:
        if old is None:sys.modules.pop('ujson',None)
        else:sys.modules['ujson']=old
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.data_root/'manifest_mirror_complete.json').read_text())
    assert manifest['complete']
    files={r['name']:r for r in manifest['files']}
    for name,row in files.items():
        path=args.data_root/name
        assert path.name==name and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        assert path.stat().st_size==row['bytes']
    # A data manifest cannot substitute executable evaluator code. Require the
    # exact separately verified publisher snapshots before importing it.
    expected_code={'hotpot_evaluate_v1.py':'d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0',
                   'squad_evaluate_v20.py':'710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4'}
    for name,digest in expected_code.items():assert files[name]['sha256']==digest
    hp=reference(args.data_root/'hotpot_evaluate_v1.py')
    sq=reference(args.data_root/'squad_evaluate_v20.py')
    rows=[];samples={};scorer_pairs=0
    for name,loader,expected in [('squad_train_v11.json',load_squad,87599),
        ('squad_dev_v11.json',load_squad,10570),('hotpot_dev_distractor_converted.json',load_hotpot,7405)]:
        options=dict(allow_unaligned_support=True) if loader is load_hotpot else {}
        tasks,gold=loader(args.data_root/name,expected_sha256=files[name]['sha256'],**options)
        assert len(tasks)==len(gold)==expected
        case_map={c.case_id:c for t in tasks for c in t.cases}
        groups={t.group for t in tasks}
        all_gold='annotations kept separate from PublicTask and Case fields'
        for task in tasks:
            assert set(task.host_task())=={'question_id','question'}
            assert all('answer' not in json.loads(c.content) and 'supporting_facts' not in json.loads(c.content) for c in task.cases)
            g=gold[task.question_id]
            # Score real annotation aliases plus deliberately wrong,empty and
            # truncated answers. No host predictions or tuned sample selection.
            predictions=(g.answers[0],'UNKNOWN','',g.answers[0].split()[0] if g.answers[0].split() else '')
            for pred in predictions:
                if loader is load_hotpot:
                    for support in (g.supporting_facts,g.supporting_facts[:1],(('absent',0),),()):
                        actual=hotpot_metrics(pred,support,g)
                        metrics={k:0. for k in ('em','f1','prec','recall','sp_em','sp_f1','sp_prec','sp_recall')}
                        _,p,r=hp.update_answer(metrics,pred,g.answers[0]);_,sp,sr=hp.update_sp(metrics,support,g.supporting_facts)
                        for key,refkey in [('em','em'),('f1','f1'),('sp_em','sp_em'),('sp_f1','sp_f1')]:
                            assert abs(actual[key]-metrics[refkey])<1e-12
                        jp,jr=p*sp,r*sr
                        assert abs(actual['joint_f1']-(2*jp*jr/(jp+jr) if jp+jr else 0.))<1e-12
                        assert actual['joint_em']==metrics['em']*metrics['sp_em']
                        scorer_pairs+=1
                else:
                    actual=answer_metrics(pred,g)
                    assert actual['em']==max(sq.compute_exact(a,pred) for a in g.answers)
                    assert abs(actual['f1']-max(sq.compute_f1(a,pred) for a in g.answers))<1e-12
                    scorer_pairs+=1
        if name=='squad_train_v11.json':
            fit=[t for t in tasks if hashlib.sha256(f'20261005\0{t.group}'.encode()).digest()[0]>=32]
            tune=[t for t in tasks if hashlib.sha256(f'20261005\0{t.group}'.encode()).digest()[0]<32]
            # Group split before question sampling prevents article/passage
            # sharing between fitting and tuning; never inspect outcomes.
            assert {t.group for t in fit}.isdisjoint({t.group for t in tune})
            samples[name]=dict(fit_ids=[t.question_id for t in select_by_id(fit,count=128,seed=20261005)],
                              tune_ids=[t.question_id for t in select_by_id(tune,count=64,seed=20261005)])
        else:
            selected=select_by_id(tasks,count=72,seed=20261005)
            samples[name]=dict(exploratory_ids=[t.question_id for t in selected[:8]],
                              reserved_ids=[t.question_id for t in selected[8:]],
                              annotation_training_allowed=False)
        gaps={qid:g.unavailable_supporting_facts for qid,g in gold.items() if g.unavailable_supporting_facts}
        rows.append(dict(name=name,questions=len(tasks),unique_cases=len(case_map),groups=len(groups),annotation_boundary=all_gold,
                         unavailable_supporting_facts=gaps,unaligned_annotations_preserved=bool(options)))
        print(name,len(tasks),len(case_map),'verified',flush=True)
    report=dict(source_fingerprint=source_fingerprint(Path(__file__).resolve().parents[1]),data=manifest,
        records=rows,scorer_pairs=scorer_pairs,samples=samples,all_checks_passed=True,
        scope='dataset/scorer engineering only; SQuAD passages and Hotpot distractor context,not fullwiki retrieval; no host/retriever outcome',
        scorer_boundary='SQuAD1.1 answers use pinned official v2 answerable EM/F1 conventions; Hotpot uses pinned v1 evaluator',
        planned_controls=['no retrieval','standard lexical one-shot','NN-kNN one-shot','NN-kNN iterative'],
        open_protocol=['exact corpus and context budget','matched generation/retrieval opportunities','need/utility training separation',
                       'frozen host/checkpoint/prompt','compute and memory accounting','specialized/calibrated head comparison'])
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('verified scorer pairs',scorer_pairs,flush=True)


if __name__=='__main__':main()

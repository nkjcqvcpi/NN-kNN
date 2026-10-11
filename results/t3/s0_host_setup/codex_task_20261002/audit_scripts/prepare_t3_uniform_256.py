import json
from pathlib import Path
import py_compile
from datetime import datetime,timezone
work=Path(__file__).resolve().parent

def rewritten(old,new,replacements):
    text=(work/old).read_text()
    for before,after in replacements:
        assert before in text,(old,before)
        text=text.replace(before,after)
    target=work/new
    with target.open('x',encoding='utf-8') as file:file.write(text)
    py_compile.compile(str(target),doraise=True)

common=[('t3_train_need_queries_freeze.json','t3_uniform_request_256_freeze.json'),
 ('t3-train-need-queries-20261005','t3-train-need-queries-256-20261005'),
 ('t3-train-need-question-control-20261005','t3-train-need-question-control-256-20261005'),
 ('t3-train-need-actual-20261005','t3-train-need-actual-256-20261005'),
 ('neural-cbr-t3-train-need-query-verification','neural-cbr-t3-train-need-query-256-verification')]
rewritten('audit_t3_train_need_queries.py','audit_t3_train_need_queries_256.py',common[:2]+common[-1:]+[
 ('len(bundle.namelist())==147','len(bundle.namelist())==149'),
 ("protocol['decode']==reference['decode']","protocol['reference_decode']==reference['decode'] and protocol['decode']==dict(reference['decode'],max_new_tokens=256)"),
 ("call['output_tokens']<=128","call['output_tokens']<=256"),
 ("artifact=json.loads((out/'need_queries.json').read_text());protocol=artifact['protocol']","artifact=json.loads((out/'need_queries.json').read_text());protocol=artifact['protocol']\nassert protocol==json.loads((out/'protocol.json').read_text()) and protocol['version']==2 and protocol['request_token_ceiling']==256")])
rewritten('replay_t3_actual_need_training.py','replay_t3_actual_need_training_256.py',common[:4]+[
 ('len(bundle.namelist())==147','len(bundle.namelist())==149'),
 ('neural-cbr-t3-actual-need-retraining-verification','neural-cbr-t3-actual-need-256-retraining-verification')])
rewritten('evaluate_t3_common_needs.py','evaluate_t3_common_needs_256.py',common[:2]+common[-1:]+[
 ('t3-train-need-{package}-20261005','t3-train-need-{package}-256-20261005'),
 ('neural-cbr-t3-common-need-evaluation','neural-cbr-t3-common-need-256-evaluation')])
rewritten('audit_t3_public_journal.py','audit_t3_public_selected_metric.py',[
 ("ap.add_argument('--expected-trials',type=int,default=80);args=ap.parse_args()","ap.add_argument('--expected-trials',type=int,default=80);ap.add_argument('--metric-checkpoint',type=Path,required=True);args=ap.parse_args()"),
 ("metric_path=work/'t3-need-matching-20261005/s8/metric.pt'","metric_path=args.metric_checkpoint\nfrom model.t3.metric_provenance import metric_provenance\nassert protocol['metric_selection']==metric_provenance(metric_path)"),
 ("sum(c['output_tokens'] for c in calls)<=384","sum(c['output_tokens'] for c in calls)<=protocol['budgets']['total_generated_token_ceiling']"),
 ("call['output_tokens']<=128","call['output_tokens']<=(protocol['request_decode']['max_new_tokens'] if call['stage']=='request' else protocol['decode']['max_new_tokens'])"),
 ("f\"need-s8-{protocol['metric_checkpoint_sha256'][:12]}\"","f\"need-s{protocol['metric_selection']['seed']}-{protocol['metric_checkpoint_sha256'][:12]}\"")])
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),freeze=json.loads((work/'t3_uniform_request_256_freeze.json').read_text()),
    initial_requests=192,request_ceiling=256,answer_continuation_ceiling=128,no_per_question_retry=True,
    fit_ids=128,tune_ids=64,seeds=[8,9,10],epochs=20,objective='hard_negative',selection='tune_recall',
    public_comparison=dict(arms=['question-control','actual'],seed=8,conditions=['learned','iterative'],
        trials_per_arm=32,scope='same existing exploratory8 SQuAD/8 Hotpot questions; reserved64 untouched',
        seed_choice='seed8 fixed before new256 public outcomes; earlier128 exploratory outcomes already known',
        boundary='exploratory redesign, not independent confirmation; separate generation/training/public costs'))
with (work/'t3_uniform_256_plan.json').open('x') as file:json.dump(plan,file,indent=2)
print('Prepared uniform256 independent audits and prospective exploratory plan')

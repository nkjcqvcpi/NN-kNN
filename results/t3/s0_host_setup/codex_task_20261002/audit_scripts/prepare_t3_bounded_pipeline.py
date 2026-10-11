import json
from pathlib import Path
import py_compile
from datetime import datetime,timezone
work=Path(__file__).resolve().parent

def write(old,new,replacements):
    text=(work/old).read_text()
    for a,b in replacements:
        assert a in text,(old,a);text=text.replace(a,b)
    target=work/new
    if target.exists():assert target.read_text()==text
    else:
        with target.open('x',encoding='utf-8') as file:file.write(text)
    if target.suffix=='.py':py_compile.compile(str(target),doraise=True)

common=[('t3_uniform_request_256_freeze.json','t3_bounded_request_256_freeze.json'),
    ('t3-train-need-queries-256-20261005','t3-train-need-queries-bounded-20261006'),
    ('t3-train-need-question-control-256-20261005','t3-train-need-question-control-bounded-20261006'),
    ('t3-train-need-actual-256-20261005','t3-train-need-actual-bounded-20261006'),
    ('neural-cbr-t3-train-need-query-256-verification','neural-cbr-t3-train-need-query-bounded-verification')]
write('audit_t3_train_need_queries_256.py','audit_t3_train_need_queries_bounded.py',common[:2]+common[-1:]+[
    ('request_prompt,validate_decision','request_prompt,validate_decision,decision_schema,decoder_schema'),
    ("protocol['format_backend']==reference['format_backend']","protocol['reference_format_backend']==reference['format_backend']\nassert protocol['request_profile']=='bounded'\nassert protocol['format_backend']['schemas']=={stage:decision_schema(stage,request_profile='bounded') for stage in ('request','answer','continuation')}\nassert protocol['format_backend']['decoder_schemas']=={stage:decoder_schema(stage,request_profile='bounded') for stage in ('request','answer','continuation')}\nfor key in ('version','backend_source_sha256'):assert protocol['format_backend'][key]==reference['format_backend'][key]"),
    ("validate_decision(json.loads(call['output'].strip()),'request')","validate_decision(json.loads(call['output'].strip()),'request',request_profile='bounded')")])
write('replay_t3_actual_need_training_256.py','replay_t3_actual_need_training_bounded.py',common[:4]+[
    ('neural-cbr-t3-actual-need-256-retraining-verification','neural-cbr-t3-actual-need-bounded-retraining-verification')])
write('evaluate_t3_common_needs_256.py','evaluate_t3_common_needs_bounded.py',common[:2]+common[-1:]+[
    ('t3-train-need-{package}-256-20261005','t3-train-need-{package}-bounded-20261006'),
    ('neural-cbr-t3-common-need-256-evaluation','neural-cbr-t3-common-need-bounded-evaluation')])
write('audit_t3_public_selected_metric.py','audit_t3_public_bounded_metric.py',[
    ("validate_decision(json.loads(raw),call['stage'])","validate_decision(json.loads(raw),call['stage'],request_profile=protocol['request_profile'])")])
text=(work/'run_t3_uniform_request_256.ps1').read_text()
start=text.index('foreach ($taskGate in')
end=text.index('Set-Location -LiteralPath',start)
text=text[:start]+'''$taskProbe = Get-CimInstance Win32_Process -Filter 'ProcessId = 53288'
if ($taskProbe) {
    if ($taskProbe.CommandLine -notlike "*& './work/t3-host-runtime/Scripts/python.exe' './work/probe_t3_bounded_requests.py'*") { throw 'Diagnostic process identity changed' }
    Write-Output 'Waiting for verified two-question diagnostic; no overlapping GPU generation.'
    Wait-Process -Id 53288
}
$taskDiagnostic = Get-Content -LiteralPath "$taskRoot/outputs/t3-bounded-request-diagnostic-20261006/summary.json" -Raw | ConvertFrom-Json
if (-not $taskDiagnostic.all_requests_complete -or $taskDiagnostic.records.Count -ne 2) { throw 'Actual bounded diagnostic incomplete' }
$taskFailed = Get-Content -LiteralPath "$taskRoot/outputs/t3-train-need-queries-256-20261005/summary.json" -Raw | ConvertFrom-Json
if ($taskFailed.calls -ne 192 -or $taskFailed.errors -ne 2) { throw 'Prior uniform256 failure not preserved' }
''' +text[end:]
for a,b in common[:4]+[
    ('t3_uniform_request_256_source_archive','t3_bounded_request_256_source_archive'),
    ('--request-token-ceiling 256','--request-token-ceiling 256 --request-profile bounded'),
    ('audit_t3_train_need_queries_256.py','audit_t3_train_need_queries_bounded.py'),
    ('replay_t3_actual_need_training_256.py','replay_t3_actual_need_training_bounded.py'),
    ('evaluate_t3_common_needs_256.py','evaluate_t3_common_needs_bounded.py'),
    ('audit_t3_public_selected_metric.py','audit_t3_public_bounded_metric.py'),
    ('t3-train-need-$taskArm-256-20261005','t3-train-need-$taskArm-bounded-20261006'),
    ('t3-public-host-need-$taskArm-256-20261005','t3-public-host-need-$taskArm-bounded-20261006'),
    ('neural-cbr-t3-public-need-$taskArm-256-verification','neural-cbr-t3-public-need-$taskArm-bounded-verification')]:
    assert a in text,a;text=text.replace(a,b)
text=text.replace('256.log','bounded.log').replace('256_audit.log','bounded_audit.log').replace('256_retraining.log','bounded_retraining.log').replace('256_evaluation.log','bounded_evaluation.log')
text=text.replace('$taskFailed = Get-Content -LiteralPath "$taskRoot/outputs/t3-train-need-queries-bounded-20261006/summary.json"', '$taskFailed = Get-Content -LiteralPath "$taskRoot/outputs/t3-train-need-queries-256-20261005/summary.json"')
with (work/'run_t3_bounded_request_256.ps1').open('x',encoding='utf-8') as file:file.write(text)
old=json.loads((work/'t3_uniform_256_plan.json').read_text())
old.update(created_utc=datetime.now(timezone.utc).isoformat(),freeze=json.loads((work/'t3_bounded_request_256_freeze.json').read_text()),request_profile='bounded',diagnostic_questions=2)
old['public_comparison']['seed_choice']='seed8 fixed before any bounded public outcomes; prior128 exploratory outcomes and128/256 train-only failures known'
with (work/'t3_bounded_request_plan.json').open('x') as file:json.dump(old,file,indent=2)
print('Prepared bounded full192 pipeline and exact grammar audits')

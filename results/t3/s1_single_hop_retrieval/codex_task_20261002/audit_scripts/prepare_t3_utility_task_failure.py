from pathlib import Path

work=Path(__file__).resolve().parent
driver=(work/'run_t3_utility_after_strict_plain.ps1').read_text()
driver=driver.replace('t3_public_set_utility_source_archive','t3_public_set_utility_task_failure_source_archive')
driver=driver.replace('t3_public_set_utility.log','t3_public_set_utility_task_failure.log')
driver=driver.replace("Write-Output 'Direct public set utility generations finished; independent outcome replay still required.'",'''& 'C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Scripts/python.exe' "$taskRoot/work/audit_t3_public_utility_task_failure.py"
if ($LASTEXITCODE -ne 0) { throw 'Independent direct utility audit failed' }
Write-Output 'Direct public set utility generations and independent recorded-outcome replay finished.' ''')
(work/'run_t3_utility_task_failure_after_strict_plain.ps1').write_text(driver,encoding='utf-8')
audit=(work/'audit_t3_public_utility.py').read_text()
audit=audit.replace('t3_public_set_utility_freeze.json','t3_public_set_utility_task_failure_freeze.json')
audit=audit.replace("            return dict(losses=None,unobserved_reason=str(exc),task_failure=True,host_call_index=global_call_index)","            score=dict(em=0.,f1=0.);loss=dict(answer=1.)\n            if key[0]=='hotpot':score.update(sp_f1=0.,joint_f1=0.);loss.update(support=1.,joint=1.)\n            return dict(losses=loss,scores=score,task_failure=True,format_error=str(exc),host_call_index=global_call_index)")
audit=audit.replace("assert old['host_calls'][-1]['stage']=='answer' and full_payload in base_prompt", "assert old['host_calls'][-1]['stage']=='answer' and base_prompt.count(full_payload)==1")
compile(audit,'audit_t3_public_utility_task_failure.py','exec')
(work/'audit_t3_public_utility_task_failure.py').write_text(audit,encoding='utf-8')
print('Prepared distinct corrected source driver/auditor; original preparation preserved.')

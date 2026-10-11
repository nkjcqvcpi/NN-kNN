from pathlib import Path
work=Path(__file__).resolve().parent
original=(work/'audit_t3_public_utility_task_failure.py').read_text()
original=original.replace("out=work.parent/'outputs/t3-public-set-utility-20261005'", "import argparse\nap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--expected-sets',type=int,required=True);args=ap.parse_args()\nout=args.output")
original=original.replace("assert len(rows)==len(originals)==protocol['planned_sets']==summary['sets']==32", "assert len(originals)==protocol['planned_sets']==32\nassert len(rows)==summary['sets']==args.expected_sets")
original=original.replace("assert {(r['dataset'],r['question_id'],r['condition']) for r in rows}==set(originals)", "assert {(r['dataset'],r['question_id'],r['condition']) for r in rows}<=set(originals) and len({(r['dataset'],r['question_id'],r['condition']) for r in rows})==len(rows)")
original=original.replace('report=dict(source=freeze,',"report=dict(provisional_snapshot=True,source=freeze,")
original=original.replace("(work.parent/'outputs/neural-cbr-t3-public-utility-verification.json').write_text", 'args.report.write_text')
compile(original,'audit_t3_public_utility_prefix.py','exec')
(work/'audit_t3_public_utility_prefix.py').write_text(original,encoding='utf-8')
print('Prepared separate partial utility verifier; terminal auditor unchanged.')

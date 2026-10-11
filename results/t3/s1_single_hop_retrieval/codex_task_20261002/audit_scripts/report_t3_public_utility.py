import hashlib
import json
from pathlib import Path

work=Path(__file__).resolve().parent;outputs=work.parent/'outputs';out=outputs/'t3-public-set-utility-20261005'
verified=json.loads((outputs/'neural-cbr-t3-public-utility-verification.json').read_text())
assert verified['all_recorded_checks_passed'] and verified['sets']==32
for name,bound in verified['input_artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==bound['sha256']
rows=[json.loads(line) for line in (out/'set_audits.jsonl').read_text().splitlines()]
summary=json.loads((out/'summary.json').read_text());groups={};pairs=[]
for dataset in ('squad','hotpot'):
    for condition in ('fixed','learned'):
        group=[r for r in rows if (r['dataset'],r['condition'])==(dataset,condition)]
        assert len(group)==8
        u=[m['loss_differences']['answer'] for r in group for m in r['audit']['marginal']]
        full=[r['audit']['outcomes'][0]['outcome']['scores']['f1'] for r in group]
        empty=[next(o for o in r['audit']['outcomes'] if o['variant']=='empty')['outcome']['scores']['f1'] for r in group]
        gains=[r['audit']['set_loss_gain']['answer'] for r in group]
        complement=[r['audit']['pair_complementarity']['answer'] for r in group]
        groups[f'{dataset}/{condition}']=dict(sets=8,pairs=16,helpful=sum(v>1e-12 for v in u),harmful=sum(v< -1e-12 for v in u),zero=sum(abs(v)<=1e-12 for v in u),
            mean_marginal=sum(u)/len(u),mean_full_f1=sum(full)/8,mean_empty_f1=sum(empty)/8,mean_set_gain=sum(gains)/8,
            mean_pair_complementarity=sum(complement)/8,positive_complementarity=sum(v>1e-12 for v in complement),
            negative_complementarity=sum(v< -1e-12 for v in complement),zero_complementarity=sum(abs(v)<=1e-12 for v in complement))
        for r in group:
            for m in r['audit']['marginal']:pairs.append(dict(dataset=dataset,condition=condition,question_id=r['question_id'],case_id=m['case_id'],utility=m['loss_differences']))
report=dict(summary=summary,verification_counts={k:verified[k] for k in ('sets','host_calls','observed_credit_pairs','invalid_outcomes','repeated_full_outputs_equal','repeated_full_outputs_different')},
    by_condition=groups,case_pairs=pairs,total_helpful=sum(g['helpful'] for g in groups.values()),total_harmful=sum(g['harmful'] for g in groups.values()),total_zero=sum(g['zero'] for g in groups.values()),
    boundary='all32 prespecified engineering sets,including source failures; direct actual full/remove/empty counterfactuals; no replacement or global truth/promotion claim; near-zero tolerance1e-12 is reporting only')
assert report['total_helpful']+report['total_harmful']+report['total_zero']==64
(outputs/'neural-cbr-t3-public-utility-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Direct public case-set utility results','',
    f"All32 actual fixed/learned sets complete:128 fresh model calls,52,730 input/3,532 output tokens,743.30s generation time;four invalid outcomes remain observed unit-loss task failures. All64 removal comparisons are observed.",'',
    '| Dataset/condition | Helpful/harmful/zero pairs | Full F1 | Empty F1 | Mean marginal | Mean full-set gain | Mean pair complementarity |','|---|---:|---:|---:|---:|---:|---:|']
for key,g in groups.items():lines.append(f"| {key} | {g['helpful']}/{g['harmful']}/{g['zero']} | {g['mean_full_f1']:.4f} | {g['mean_empty_f1']:.4f} | {g['mean_marginal']:+.4f} | {g['mean_set_gain']:+.4f} | {g['mean_pair_complementarity']:+.4f} |")
lines+=['',f"Total64 conditional pairs:helpful{report['total_helpful']},harmful{report['total_harmful']},zero{report['total_zero']}. Positive means removing the case worsens the declared task loss;negative means removing it improves this host/query/set outcome. Zero does not establish falsehood or irrelevance.",
    'All32 fresh full-set outputs exactly equal their archived source outputs. Every subset prompt differs only by its recorded evidence removal;no refill occurs.128 call/outcome journals,objective scores,activation exposure,C/H/Q and complementarity pass independent recorded-output/manual-arithmetic replay.',
    '',report['boundary'],'',
    'Evidence:outputs/t3-public-set-utility-20261005/;outputs/neural-cbr-t3-public-utility-verification.json;outputs/neural-cbr-t3-public-utility-results.json.',
    'Source144 files:aaba914ea1445c77bcb474a0cb0f18578460a372905f02cafe9222746afee797;ZIP62defc4cca7082cb471fd3741cdf3e43a22b8261f7d87bc8f638f7e2bbb3bb2c.']
(outputs/'neural-cbr-t3-public-utility-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'case_pairs','boundary'}}))

"""Recorded ablation: query-aware NN-CDH without the retrieval prediction input."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from model.common import tracking
from model.t1 import records
from tools import t1_query_aware as q

EXPERIMENT="nopred_v1"
SOURCE_STUDY="aq1"
DESIGN="query_aware_no_prediction"
METHOD="query_no_prediction"
METHODS=("retrieval_only","original","query_aware",METHOD,*q.BASELINES)
PROTOCOL=dict(version=1,adapter_design=DESIGN,input="difference_and_query_only",
    initialization="remove_prediction_columns_preserve_other_weights_and_rng",
    training="original_am1_F_schedule_from_pretrained_core",reference_study=SOURCE_STUDY,
    interventions=["rotate_context","zero_context","zero_query_channel","rotate_prediction_only"])


def key_for(study,task,seed):
    return records.key(study,EXPERIMENT,task.split("__",1)[1],"nopred",f"s{seed}")


def reference(task,seed):
    refs=q.source_refs(task,seed)
    _,source,core,_,fp=refs
    unit=tracking.load_unit(q.key_for(SOURCE_STUDY,task,seed,"query_aware"),
        config=q.config_for(task,seed,"query_aware",source,core,fp),arrays=False)
    if unit is None:
        raise ValueError("the completed query-aware reference is required")
    return refs,unit


def config_for(task,seed,refs,unit):
    _,source,core,_,fp=refs
    return dict(experiment=EXPERIMENT,protocol=PROTOCOL,task=task,seed=seed,
        source_unit=source.info.run_id,source_config_identity=tracking.config_identity(source.config),
        core_unit=core.info.run_id,core_config_identity=tracking.config_identity(core.config),
        query_reference=unit.info.run_id,query_config_identity=tracking.config_identity(unit.config),
        data_fingerprint=fp)


def load_data(task,seed,refs):
    _,source,core,_,fp=refs
    unit=tracking.load_unit(q.key_for(SOURCE_STUDY,task,seed,"data"),
        config=q.config_for(task,seed,"data",source,core,fp),arrays=False)
    if unit is None:
        raise ValueError("the recorded aq1 split is required")
    data=records.unpack(unit.load_state("data"))
    if q.data_fingerprint(data)!=fp:
        raise ValueError("cached split fingerprint mismatch")
    return data


@torch.no_grad()
def prediction_only_probe(adapter,kind,values,target):
    normal=q.predict_context(adapter,kind,values)
    rotated={**values,"p0":values["p0"].roll(1,0)}
    changed=q.predict_context(adapter,kind,rotated)
    correction=normal-values["p0"]
    other=changed-rotated["p0"]
    torch.testing.assert_close(correction,other,atol=2e-6,rtol=1e-5)
    shift=rotated["p0"]-values["p0"]
    torch.testing.assert_close(changed-normal,shift,atol=2e-6,rtol=1e-5)
    difference=changed-normal
    if kind=="classification": difference=difference-difference.mean(1,keepdim=True)
    rms=float(difference.square().mean().sqrt())
    prms=float(shift.square().mean().sqrt())
    row=dict(intervention="rotate_prediction_only",error=q.metrics(changed,target,kind)["error"],
        baseline_error=q.metrics(normal,target,kind)["error"],output_rms_change=rms,retrieval_rms_change=prms,
        residual_max_change=float((correction-other).abs().max()),
        translation_max_error=float((changed-normal-shift).abs().max()),
        sensitivity_ratio=rms/prms if kind=="regression" and prms>1e-12 else None)
    if kind=="classification": row["class_change_rate"]=float((changed.argmax(1)!=normal.argmax(1)).float().mean())
    return row,changed


def run_one(study,task,seed):
    refs,old_query=reference(task,seed)
    config=config_for(task,seed,refs,old_query)
    key=key_for(study,task,seed)
    if tracking.load_unit(key,config=config,arrays=False) is not None:
        print(f"[recorded] {task} s{seed} {METHOD}",flush=True)
        return
    data=load_data(task,seed,refs)
    _,source,core,_,_=refs
    with tracking.open_run(key,config=config) as run:
        model,adapter,desc,info=q.fit_from_source(core,source.load_state("checkpoint"),data,run,DESIGN)
        row=dict(task=task,task_type=data.task_type,seed=seed,method=METHOD,
            selected_epoch=info["best_epoch"],epochs_run=len(info["history"]),
            adapter_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad))
        for split in ("val","test"):
            target=getattr(data,"y_"+split)
            ref=old_query.load_state(split+"_predictions")
            if not torch.equal(ref["target"],target): raise AssertionError("reference targets differ")
            values=q.context(model,getattr(data,"X_"+split))
            final=q.predict_context(adapter,data.task_type,values)
            row.update({split+"_"+k:v for k,v in q.metrics(final,target,data.task_type,values["p0"]).items()})
            run.save_state(split+"_predictions",records.storable(dict(**values,final=final,target=target)))
        effects,outputs=q.intervention_results(adapter,data.task_type,values,data.y_test)
        extra,changed=prediction_only_probe(adapter,data.task_type,values,data.y_test)
        effects.append(extra); outputs[extra["intervention"]]=changed
        for effect in effects: effect.update(task=task,task_type=data.task_type,seed=seed,method=METHOD)
        run.log_records("interventions",[records.ordered(e) for e in effects])
        run.save_state("intervention_predictions",records.storable(dict(outputs=outputs,target=data.y_test)))
        run.save_state("checkpoint",records.storable(dict(model_state=model.state_dict(),adapter_state=adapter.state_dict(),
            model_description=desc,training_info=info)))
        run.log_record("history",records.ordered(info["history"]))
        run.log_records("rows",[records.ordered(row)])
        run.log_final({k:v for k,v in row.items() if isinstance(v,(int,float))})
        records.save_rng(run)
    print(f"[done] {task} s{seed} {METHOD}: error={row['test_error']:.6f}, epoch={row['selected_epoch']}",flush=True)


def ratio_summary(pairs):
    differences=[a["test_error"]-b["test_error"] for a,b in pairs]
    positive=[(a,b) for a,b in pairs if min(a["test_error"],b["test_error"])>0]
    return dict(n_tasks=len(pairs),wins=sum(d < -1e-7 for d in differences),
        ties=sum(abs(d)<=1e-7 for d in differences),losses=sum(d>1e-7 for d in differences),
        gmr=float(np.exp(np.mean([np.log(a["test_error"]/b["test_error"]) for a,b in positive]))) if positive else None,
        zero_exclusions=[a["task"] for a,b in pairs if min(a["test_error"],b["test_error"])<=0])


def summarize(study,tasks,seeds):
    baseline=q.summarize(SOURCE_STUDY,tasks,seeds,write=False)
    if baseline["missing"]: raise ValueError("query-aware reference study is incomplete")
    rows,effects=list(baseline["rows"]),list(baseline["interventions"])
    missing=[]
    regression_behavior=[]
    harm_examples=[]
    prediction_checks=0
    for task in tasks:
        for seed in seeds:
            refs,old_query=reference(task,seed)
            key=key_for(study,task,seed)
            unit=tracking.load_unit(key,config=config_for(task,seed,refs,old_query),arrays=False)
            if unit is None: missing.append(key.run_id); continue
            row,=records.rows(unit,"rows")
            kind=row["task_type"]
            for split in ("val","test"):
                saved=unit.load_state(split+"_predictions")
                if not torch.equal(saved["target"],old_query.load_state(split+"_predictions")["target"]):
                    raise AssertionError("paired targets differ")
                metrics=q.metrics(saved["final"],saved["target"],kind,saved["p0"])
                for name,value in metrics.items():
                    if not np.isclose(value,row[split+"_"+name],atol=2e-6,rtol=1e-5):
                        raise AssertionError("new prediction metric mismatch")
                prediction_checks+=1
            stored=unit.load_state("intervention_predictions")
            description=unit.load_state("checkpoint")["model_description"]
            options=description.get("sync",description.get("adapter"))
            old_row,=records.rows(old_query,"rows")
            removed=saved["p0"].shape[1]*options["hidden_dims"][0]
            if row["adapter_parameters"]!=old_row["adapter_parameters"]-removed:
                raise AssertionError("prediction input parameter count mismatch")
            row["removed_parameters"]=removed
            if kind=="regression":
                from tools.t1_adaptation_diagnostics import regression_diagnostics
                query_saved=old_query.load_state("test_predictions")
                original_saved=old_query.load_state("intervention_predictions")
                if not torch.equal(original_saved["target"],saved["target"]):
                    raise AssertionError("original diagnostic targets differ")
                variants=[("original",original_saved["original_context"]["p0"],original_saved["original_normal"]),
                          ("query_aware",query_saved["p0"],query_saved["final"]),(METHOD,saved["p0"],saved["final"])]
                for method,pre,post in variants:
                    stats,cases,_=regression_diagnostics(pre.numpy().reshape(-1),post.numpy().reshape(-1),saved["target"].numpy().reshape(-1))
                    ranked=sorted(enumerate(cases),key=lambda x:x[1]["squared_error_change"],reverse=True)
                    stats["largest_harm_share"]=max(0.,ranked[0][1]["squared_error_change"])/stats["total_harm_sse"] if stats["total_harm_sse"]>0 else 0.
                    for index,case in ranked[:5]:
                        if case["squared_error_change"]<=0: continue
                        harm_examples.append(dict(task=task,seed=seed,method=method,query_index=index,
                            retrieval=float(pre.reshape(-1)[index]),final=float(post.reshape(-1)[index]),
                            target=float(saved["target"].reshape(-1)[index]),**case))
                    regression_behavior.append(dict(task=task,seed=seed,method=method,
                        **{k:v for k,v in stats.items() if not k.startswith("oracle_")}))
            for effect in records.rows(unit,"interventions"):
                changed=stored["outputs"][effect["intervention"]]
                actual=q.metrics(changed,stored["target"],kind)["error"]
                if not np.isclose(actual,effect["error"],atol=2e-6,rtol=1e-5):
                    raise AssertionError("intervention error mismatch")
                if effect["intervention"]=="rotate_prediction_only":
                    shift=saved["p0"].roll(1,0)-saved["p0"]
                    torch.testing.assert_close(changed-saved["final"],shift,atol=2e-6,rtol=1e-5)
                    torch.testing.assert_close(changed-saved["p0"].roll(1,0),saved["final"]-saved["p0"],atol=2e-6,rtol=1e-5)
                    residual_error=float(((changed-saved["p0"].roll(1,0))-(saved["final"]-saved["p0"])).abs().max())
                    translation_error=float((changed-saved["final"]-shift).abs().max())
                    if residual_error!=effect["residual_max_change"] or translation_error!=effect["translation_max_error"]:
                        raise AssertionError("prediction-only invariance diagnostic mismatch")
                prediction_checks+=1
                effects.append(effect)
            rows.append(row)
    if len({(r["task"],r["seed"],r["method"]) for r in rows})!=len(rows): raise AssertionError("duplicate rows")
    task_rows=[]
    for task in tasks:
        for method in METHODS:
            part=[r for r in rows if r["task"]==task and r["method"]==method]
            if len(part)!=len(seeds): continue
            item=dict(task=task,task_type=part[0]["task_type"],method=method,n_seeds=len(part),
                test_error=float(np.mean([r["test_error"] for r in part])))
            for field in ("test_pre_error","test_helped","test_harmed","test_n"):
                if all(field in r for r in part):
                    item[field]=float(np.mean([r[field] for r in part])) if field.endswith("error") else sum(r[field] for r in part)
            task_rows.append(item)
    families,comparisons,adaptation,intervention_summary=[],[],[],[]
    for kind in ("classification","regression"):
        for method in METHODS:
            part=[r for r in task_rows if r["task_type"]==kind and r["method"]==method]
            if part:
                families.append(dict(task_type=kind,method=method,n_tasks=len(part),mean_error=float(np.mean([r["test_error"] for r in part]))))
        for ref in ("original","query_aware",*q.BASELINES):
            pairs=[]
            for a in task_rows:
                if a["task_type"]==kind and a["method"]==METHOD:
                    b=next(r for r in task_rows if r["task"]==a["task"] and r["method"]==ref)
                    pairs.append((a,b))
            if pairs: comparisons.append(dict(task_type=kind,reference=ref,**ratio_summary(pairs)))
        for method in ("original","query_aware",METHOD):
            part=[r for r in rows if r["task_type"]==kind and r["method"]==method and "test_helped" in r]
            if part:
                adaptation.append(dict(task_type=kind,method=method,n_runs=len(part),
                    mean_pre_error=float(np.mean([r["test_pre_error"] for r in part])),
                    mean_final_error=float(np.mean([r["test_error"] for r in part])),
                    helped=sum(r["test_helped"] for r in part),harmed=sum(r["test_harmed"] for r in part),n=sum(r["test_n"] for r in part)))
            for intervention in PROTOCOL["interventions"]:
                part=[r for r in effects if r["task_type"]==kind and r["method"]==method and r["intervention"]==intervention]
                if not part: continue
                item=dict(task_type=kind,method=method,intervention=intervention,n_runs=len(part))
                for field in ("error","baseline_error","output_rms_change","retrieval_rms_change","sensitivity_ratio","class_change_rate"):
                    values=[r[field] for r in part if r.get(field) is not None]
                    if values: item["mean_"+field]=float(np.mean(values))
                if intervention=="rotate_prediction_only":
                    item["max_residual_change"]=max(r["residual_max_change"] for r in part)
                    item["max_translation_error"]=max(r["translation_max_error"] for r in part)
                intervention_summary.append(item)
    report=dict(study=study,protocol=PROTOCOL,missing=missing,prediction_checks=prediction_checks,
        reference_prediction_checks=baseline["prediction_checks"],rows=rows,tasks=task_rows,families=families,
        comparisons=comparisons,adaptation=adaptation,interventions=effects,intervention_summary=intervention_summary,
        regression_behavior=regression_behavior,harm_examples=harm_examples)
    out=REPO/"results"/"adaptation_improvement"/study
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name,items in (("runs",rows),("tasks",task_rows),("families",families),("comparisons",comparisons),
                       ("adaptation",adaptation),("interventions",effects),("intervention_summary",intervention_summary),
                       ("regression_behavior",regression_behavior),("harm_examples",harm_examples)):
        if not items: continue
        with (out/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)))
            writer.writeheader(); writer.writerows(items)
    if not missing: plot_summary(report,out)
    print(json.dumps(dict(rows=len(rows),missing=len(missing),prediction_checks=prediction_checks,
        reference_prediction_checks=baseline["prediction_checks"],summary=str(out/"comparison.json"))),flush=True)
    return report


def plot_summary(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=("original","query_aware",METHOD)
    names=("Original residual", "Query + retrieved prediction", "Query, no prediction input")
    kinds=[k for k in ("classification","regression") if any(r["task_type"]==k for r in report["families"])]
    fig,axes=plt.subplots(1,len(kinds),figsize=(7*len(kinds),3.6),squeeze=False,sharey=True,layout="constrained")
    for ax,kind in zip(axes[0],kinds):
        family={r["method"]:r for r in report["families"] if r["task_type"]==kind}
        values=[family[m]["mean_error"]*(100 if kind=="classification" else 1) for m in methods]
        ax.barh(names,values,color=("#8b9eaf","#637c93","#24797d"))
        ax.set_ylim(2.5,-.5); ax.set_xlim(0,max(values)*1.25)
        ax.set_title(kind.capitalize()+f" / {family[METHOD]['n_tasks']} tasks")
        ax.set_xlabel("Mean classification error (%)" if kind=="classification" else "Mean standardized RMSE")
        for i,value in enumerate(values): ax.text(value,i,"  "+(f"{value:.3f}" if kind=="classification" else f"{value:.6f}"),va="center")
    fig.suptitle("Removing the retrieved prediction from the residual network")
    fig.savefig(out/"query_no_prediction_comparison.png",dpi=170)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--mode",choices=("run","summarize"),default="run")
    parser.add_argument("--tasks",nargs="+",default=q.prior.TASKS)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(q.prior.SEEDS))
    args=parser.parse_args(argv)
    if not args.tasks or len(set(args.tasks))!=len(args.tasks) or not set(args.tasks)<=set(q.prior.TASKS): parser.error("choose unique local tasks")
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(q.prior.SEEDS): parser.error("choose unique seeds 40-42")
    if args.study in ("am1","ag1","ag2","aq1"): parser.error("use a new study")
    records.check_study(args.study)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode=="summarize": return summarize(args.study,args.tasks,args.seeds)
    records.check_unit_paths(args.study,[(EXPERIMENT,t.split("__",1)[1],"nopred",f"s{s}") for t in args.tasks for s in args.seeds])
    for task in args.tasks:
        for seed in args.seeds: run_one(args.study,task,seed)


if __name__=="__main__": main()

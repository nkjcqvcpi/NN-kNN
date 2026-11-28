"""Recorded query-aware original NN-CDH study and matched local baselines."""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
os.environ.setdefault("NNKNN_DEVICE","cpu")

import numpy as np
import torch
from model.common import tracking
from model.common.core import CoreConfig, build_model, clone_optimizer
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter
from model.t1 import records
from model.t1.calibration import FreeRadius
from model.t1.outcomes import final_prediction
from model.t1.query_aware import is_query_aware
from model.t1.reuse import ReuseConfig, train_regression_adapter
from model.t1.sync import SyncConfig, train_synchronized
from tools import t1_adaptation_gate as prior
from tools.t1_adaptation_improvement import data_fingerprint
from tools.t1_grid_search import fit_candidate, prediction, numeric_metrics

EXPERIMENT="query_v1"
PROTOCOL=dict(version=1, adapter="original_residual_plus_query", query_initialization="zero_new_connections",
              nnknn_budget="original_am1_F_schedule", source_study="am1", baseline_epochs=120,
              baseline_patience=30, baseline_selection="validation_error_then_loss_then_candidate_id",
              interventions=["rotate_context","zero_context","zero_query_channel"])
BASELINES={
    "mlp":[dict(width=w,layers=2,lr=.001,weight_decay=0.,dropout=0.) for w in (64,128)],
    "tabm":[dict(d_block=w,n_blocks=2,k=8,lr=.001,weight_decay=0.) for w in (64,128)],
    "catboost":[dict(depth=d,learning_rate=.1,l2_leaf_reg=3.,random_strength=1.,iterations=300) for d in (4,8)],
    "knn":[dict(n_neighbors=k,weights="distance",p=2) for k in (3,5,10,20)],
}
METHODS=("retrieval_only","original","query_aware","knn","mlp","tabm","catboost")


def key_for(study,task,seed,arm):
    return records.key(study,EXPERIMENT,task.split("__",1)[1],arm,f"s{seed}")


def source_refs(task,seed):
    cfg,probe,source,fp=prior.source_for(task,seed,"am1")
    ds=cfg["datasets"][0]
    key=probe.unit_key(ds,"_core__tuned",seed)
    core=tracking.load_unit(key,config=probe.unit_config(probe.run_config(ds,seed,{"stage":"core__tuned"})),arrays=False)
    ro,_=probe.pair_units(ds)
    retrieval=tracking.load_unit(probe.unit_key(ds,ro[0],seed),
        config=probe.unit_config(probe.run_config(ds,seed,ro[1])),arrays=False)
    if core is None or retrieval is None:
        raise ValueError("complete source core and retrieval-only units are required")
    return probe,source,core,retrieval,fp


def config_for(task,seed,arm,source,core,fp,**extra):
    return dict(experiment=EXPERIMENT,task=task,seed=seed,arm=arm,protocol=PROTOCOL,
                source_unit=source.info.run_id,source_config_identity=tracking.config_identity(source.config),
                core_unit=core.info.run_id,core_config_identity=tracking.config_identity(core.config),
                data_fingerprint=fp,**extra)


def recorded_data(study,task,seed,refs):
    probe,source,core,_,fp=refs
    config=config_for(task,seed,"data",source,core,fp)
    key=key_for(study,task,seed,"data")
    saved=tracking.load_unit(key,config=config,arrays=False)
    if saved is not None:
        return records.unpack(saved.load_state("data"))
    data=probe.data_for(probe.cfg["datasets"][0],seed)
    if data_fingerprint(data)!=fp:
        raise ValueError("data differ from the original paired experiment")
    with tracking.open_run(key,config=config) as run:
        run.save_state("data",records.pack(data))
        run.log_record("data",dict(fingerprint=fp,n_train=len(data.y_train),n_val=len(data.y_val),n_test=len(data.y_test)))
        records.save_rng(run)
    return data


@torch.no_grad()
def context(model,X):
    model.eval()
    parts=[]
    for x in X.split(128):
        r=model.retrieve(x,exclude_identical=False)
        w=r["weights"]
        p0=w@model.labels[r["case_indices"]].float()
        if model.task_type=="classification":
            p0=p0/p0.sum(1,keepdim=True).clamp_min(1e-12)
        parts.append(dict(query=r["query_features"],aggregate=w@r["case_features"],p0=p0))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def predict_context(adapter,kind,c,*,query_channel=None):
    adapter.eval()
    dz,p0=c["query"]-c["aggregate"],c["p0"]
    query=c["query"] if query_channel is None else query_channel
    aware=is_query_aware(adapter)
    if kind=="classification":
        return adapter(dz,p0,**({"query_features":query} if aware else {}))[1]
    if aware:
        return adapter.forward_differences(dz,p0,query)
    return p0+adapter.adapt_net_final(torch.cat((dz,p0),1))


def metrics(scores,target,kind,p0=None):
    if kind=="classification":
        wrong=scores.argmax(1)!=target.long()
        error=float(wrong.float().mean())
        before=p0.argmax(1)!=target.long() if p0 is not None else wrong
        helped,harmed=before&~wrong,~before&wrong
    else:
        after=(scores-target.reshape_as(scores)).square().mean(1)
        before=(p0-target.reshape_as(p0)).square().mean(1) if p0 is not None else after
        error=float(after.mean().sqrt())
        helped,harmed=after<before-1e-8,after>before+1e-8
    result=dict(n=len(target),error=error,helped=int(helped.sum()),harmed=int(harmed.sum()))
    if p0 is not None:
        result["pre_error"]=metrics(p0,target,kind)["error"]
    return result


@torch.no_grad()
def intervention_results(adapter,kind,c,target):
    normal=predict_context(adapter,kind,c)
    base=metrics(normal,target,kind)["error"]
    variants={"rotate_context":{**c,"aggregate":c["aggregate"].roll(1,0),"p0":c["p0"].roll(1,0)},
              "zero_context":{**c,"aggregate":torch.zeros_like(c["aggregate"]),
                   "p0":torch.full_like(c["p0"],1/c["p0"].shape[1]) if kind=="classification" else torch.zeros_like(c["p0"])}}
    if is_query_aware(adapter):
        variants["zero_query_channel"]=c
    results,outputs=[],{}
    for name,v in variants.items():
        changed=predict_context(adapter,kind,v,query_channel=torch.zeros_like(c["query"]) if name=="zero_query_channel" else None)
        difference=changed-normal
        if kind=="classification":
            difference=difference-difference.mean(1,keepdim=True)
        rms=float(difference.square().mean().sqrt())
        retrieval_rms=float((v["p0"]-c["p0"]).square().mean().sqrt())
        item=dict(intervention=name,error=metrics(changed,target,kind)["error"],baseline_error=base,
                  output_rms_change=rms,retrieval_rms_change=retrieval_rms,
                  sensitivity_ratio=rms/retrieval_rms if retrieval_rms>1e-12 and kind=="regression" else None)
        if kind=="classification":
            item["class_change_rate"]=float((changed.argmax(1)!=normal.argmax(1)).float().mean())
        results.append(item)
        outputs[name]=changed
    return results,outputs


def rebuild_original(state,data):
    desc=state["model_description"]
    labels=state["model_state"]["labels"]
    cc=replace(CoreConfig(**desc["core"]),bias_init="manual")
    y=labels.argmax(1) if data.task_type=="classification" else labels.flatten()
    model=build_model(state["model_state"]["cases"],y,cc,data.num_classes)
    model.load_state_dict(state["model_state"])
    model.eval()
    dim=model.retrieve(data.X_val[:1],exclude_identical=False)["query_features"].shape[1]
    options=desc.get("sync",desc.get("adapter"))
    if data.task_type=="classification":
        adapter=ClassificationNNCDHAdapter(dim,data.num_classes,hidden_dims=tuple(options["hidden_dims"]),
            output_mode=options["output_mode"],dropout=options.get("adapter_dropout",options.get("dropout",0.)))
    else:
        adapter=NNCDHAdapter(dim,1,tuple(options["hidden_dims"]))
    adapter.load_state_dict(state["adapter_state"])
    return model,adapter.eval()


def fit_from_source(core,original,data,run,design):
    """Start a residual input ablation from the same recorded pretraining stage."""
    base=records.unpack(core.load_state("objects"))
    records.restore_rng_of(core)
    model,cc,tr=base["model"],base["cc"],base["tr"]
    desc=copy.deepcopy(original["model_description"])
    if "sync" in desc:
        cfg=SyncConfig(**{**desc["sync"],"adapter_design":design})
        if not desc["core_pretrain"] or cfg.maintains:
            raise ValueError("matched query experiment expects a pretrained core without maintenance")
        optimizer=clone_optimizer(tr.optimizer,model,cc)
        for group in optimizer.param_groups:
            group["lr"]*=desc["retrieval_lr_scale"]
        radius=FreeRadius(**desc["free_radius_at_t_star"])
        adapter,info=train_synchronized(model,data,cc,cfg,radius,near_scale=desc["near_scale"],
                                       core_optimizer=optimizer,run=run)
        desc["sync"],desc["free_radius_selected"]=cfg.__dict__,info["free_radius"]
    else:
        cfg=ReuseConfig(**{**desc["adapter"],"adapter_design":design})
        adapter,info=train_regression_adapter(model,data,cfg,run=run)
        desc["adapter"]=cfg.__dict__
    model.eval(); adapter.eval()
    return model,adapter,desc,info


def run_query(study,task,seed,refs,data):
    _,source,core,_,fp=refs
    config=config_for(task,seed,"query_aware",source,core,fp)
    key=key_for(study,task,seed,"query_aware")
    if tracking.load_unit(key,config=config,arrays=False) is not None:
        print(f"[recorded] {task} s{seed} query_aware",flush=True)
        return
    with tracking.open_run(key,config=config) as run:
        original=source.load_state("checkpoint")
        model,adapter,desc,info=fit_from_source(core,original,data,run,"query_aware")
        row=dict(task=task,task_type=data.task_type,seed=seed,method="query_aware",
                 selected_epoch=info["best_epoch"],epochs_run=len(info["history"]),
                 adapter_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad))
        for split in ("val","test"):
            values=context(model,getattr(data,"X_"+split))
            final=predict_context(adapter,data.task_type,values)
            target=getattr(data,"y_"+split)
            row.update({split+"_"+k:v for k,v in metrics(final,target,data.task_type,values["p0"]).items()})
            run.save_state(split+"_predictions",records.storable(dict(**values,final=final,target=target)))
        effects,outputs=intervention_results(adapter,data.task_type,values,data.y_test)
        original_model,original_adapter=rebuild_original(original,data)
        old_context=context(original_model,data.X_test)
        old_output=predict_context(original_adapter,data.task_type,old_context)
        events=sorted(records.rows(source,"retrieval_events"),key=lambda e:e["query_index"])
        expected=torch.tensor([e["final_values"] if data.task_type=="classification" else [e["final_prediction"]] for e in events])
        if not torch.allclose(old_output,expected,atol=2e-6,rtol=1e-5):
            raise AssertionError("original predictions did not reconstruct")
        old_effects,old_outputs=intervention_results(original_adapter,data.task_type,old_context,data.y_test)
        for method,items in (("query_aware",effects),("original",old_effects)):
            for item in items: item.update(task=task,task_type=data.task_type,seed=seed,method=method)
        run.log_records("interventions",[records.ordered(r) for r in effects+old_effects])
        run.save_state("intervention_predictions",records.storable(dict(query_aware=outputs,original=old_outputs,
            original_normal=old_output,original_context=old_context,target=data.y_test)))
        run.save_state("checkpoint",records.storable(dict(model_state=model.state_dict(),adapter_state=adapter.state_dict(),
            model_description=desc,training_info=info)))
        run.log_record("history",records.ordered(info["history"]))
        run.log_records("rows",[records.ordered(row)])
        run.log_final({k:v for k,v in row.items() if isinstance(v,(int,float))})
        records.save_rng(run)
    print(f"[done] {task} s{seed} query_aware: error={row['test_error']:.6f}, epoch={row['selected_epoch']}",flush=True)


def run_baseline(study,task,seed,family,refs,data):
    _,source,core,_,fp=refs
    base_config=config_for(task,seed,family,source,core,fp,candidates=BASELINES[family])
    final_key=key_for(study,task,seed,family)
    if tracking.load_unit(final_key,config=base_config,arrays=False) is not None:
        print(f"[recorded] {task} s{seed} {family}",flush=True)
        return
    allowed=dict(X_train=data.X_train,y_train=data.y_train,X_val=data.X_val,y_val=data.y_val,
                 task=data.task_type,classes=data.num_classes)
    candidates=[]
    for index,params in enumerate(BASELINES[family]):
        arm=f"{family}_c{index}"
        conf=config_for(task,seed,arm,source,core,fp,parameters=params)
        key=key_for(study,task,seed,arm)
        saved=tracking.load_unit(key,config=conf,arrays=False)
        if saved is None:
            with tracking.open_run(key,config=conf) as run:
                fitted,info=fit_candidate(family,params,allowed,seed,PROTOCOL["baseline_epochs"],PROTOCOL["baseline_patience"],run=run)
                val=prediction(fitted,family,data.X_val,data.task_type)
                result=numeric_metrics(val,data.y_val.numpy(),data.task_type,data.y_scale)
                error=1-result["accuracy"] if data.task_type=="classification" else result["rmse_standardized"]
                run.log_record("candidate",dict(candidate=index,validation_error=error,validation_loss=result["loss"],info=records.jsonable(info)))
                run.save_state("model",records.pack(fitted))
                run.save_state("validation_predictions",records.storable(dict(prediction=val,target=data.y_val)))
                run.log_record("history",records.ordered(info.get("history",[])))
                records.save_rng(run)
            saved=tracking.load_unit(key,config=conf,arrays=False)
        record=records.record(saved,"candidate")
        candidates.append((record["validation_error"],record["validation_loss"],index,saved))
    winner=min(candidates,key=lambda x:x[:3])
    with tracking.open_run(final_key,config=base_config) as run:
        fitted=records.unpack(winner[3].load_state("model"))
        pred=prediction(fitted,family,data.X_test,data.task_type)
        result=numeric_metrics(pred,data.y_test.numpy(),data.task_type,data.y_scale)
        error=1-result["accuracy"] if data.task_type=="classification" else result["rmse_standardized"]
        row=dict(task=task,task_type=data.task_type,seed=seed,method=family,test_error=error,test_n=len(data.y_test),
                 val_error=winner[0],selected_candidate=winner[2],selected_unit=winner[3].info.run_id)
        run.log_records("rows",[records.ordered(row)])
        run.log_final({"test_error":error,"validation_error":winner[0]})
        run.save_state("test_predictions",records.storable(dict(prediction=pred,target=data.y_test)))
        run.save_state("model_reference",dict(unit=winner[3].info.run_id))
        records.save_rng(run)
    print(f"[done] {task} s{seed} {family}: error={error:.6f}, candidate={winner[2]}",flush=True)


def run_task(study,task,seeds,mode):
    arms=() if mode=="data" else ("query_aware",) if mode=="query" else tuple(BASELINES) if mode=="baselines" else ("query_aware",*BASELINES)
    records.check_unit_paths(study,[(EXPERIMENT,task.split("__",1)[1],arm,f"s{s}")
        for s in seeds for arm in ("data",*arms,*[f"{f}_c{i}" for f,cs in BASELINES.items() for i in range(len(cs))])])
    for seed in seeds:
        refs=source_refs(task,seed)
        data=recorded_data(study,task,seed,refs)
        for arm in arms:
            if arm=="query_aware": run_query(study,task,seed,refs,data)
            else: run_baseline(study,task,seed,arm,refs,data)


def summarize(study,tasks,seeds,*,write=True):
    rows,effects,missing=[],[],[]
    prediction_checks=0
    for task in tasks:
        for seed in seeds:
            _,source,core,retrieval,fp=source_refs(task,seed)
            for method,unit,output in (("original",source,"post"),("retrieval_only",retrieval,"pre")):
                r=records.record(unit,"row")
                error=1-r[f"test_accuracy_{output}"] if task.startswith("classification") else r[f"test_rmse_{output}"]
                rows.append(dict(task=task,task_type=task.split("__")[0],seed=seed,method=method,test_error=error,
                    test_pre_error=(1-r["test_accuracy_pre"] if task.startswith("classification") else r["test_rmse_pre"])))
            for method in ("query_aware",*BASELINES):
                config=config_for(task,seed,method,source,core,fp,**({"candidates":BASELINES[method]} if method in BASELINES else {}))
                key=key_for(study,task,seed,method)
                unit=tracking.load_unit(key,config=config,arrays=False)
                if unit is None: missing.append(key.run_id); continue
                row,=records.rows(unit,"rows")
                saved=unit.load_state("test_predictions")
                output=saved["final"] if method=="query_aware" else torch.as_tensor(saved["prediction"])
                actual=metrics(output.reshape(-1,1) if output.ndim==1 else output,saved["target"],row["task_type"])["error"]
                if not np.isclose(actual,row["test_error"],atol=2e-6,rtol=1e-5): raise AssertionError("prediction metric mismatch")
                prediction_checks+=1
                if method=="query_aware":
                    for split in ("val","test"):
                        recorded=unit.load_state(split+"_predictions")
                        calculated=metrics(recorded["final"],recorded["target"],row["task_type"],recorded["p0"])
                        for field,value in calculated.items():
                            if not np.isclose(value,row[split+"_"+field],atol=2e-6,rtol=1e-5):
                                raise AssertionError("query-aware correction diagnostic mismatch")
                    intervention=unit.load_state("intervention_predictions")
                    old_metrics=metrics(intervention["original_normal"],intervention["target"],row["task_type"],
                                        intervention["original_context"]["p0"])
                    original_row=next(r for r in rows if r["task"]==task and r["seed"]==seed and r["method"]=="original")
                    for field in ("error","pre_error"):
                        if not np.isclose(old_metrics[field],original_row["test_"+field],atol=2e-6,rtol=1e-5):
                            raise AssertionError("original intervention reference differs from source metrics")
                    original_row.update({"test_"+k:v for k,v in old_metrics.items() if k not in ("error","pre_error")})
                    for effect in records.rows(unit,"interventions"):
                        actual=metrics(intervention[effect["method"]][effect["intervention"]],intervention["target"],row["task_type"])["error"]
                        if not np.isclose(actual,effect["error"],atol=2e-6,rtol=1e-5):
                            raise AssertionError("intervention metric mismatch")
                        prediction_checks+=1
                else:
                    options=[]
                    for index,params in enumerate(BASELINES[method]):
                        arm=f"{method}_c{index}"
                        candidate=tracking.load_unit(key_for(study,task,seed,arm),
                            config=config_for(task,seed,arm,source,core,fp,parameters=params),arrays=False)
                        if candidate is None: raise AssertionError("selected family has a missing candidate")
                        receipt=records.record(candidate,"candidate")
                        vp=candidate.load_state("validation_predictions")
                        calculated=numeric_metrics(np.asarray(vp["prediction"]),np.asarray(vp["target"]),row["task_type"])
                        val_error=1-calculated["accuracy"] if row["task_type"]=="classification" else calculated["rmse_standardized"]
                        if not np.isclose(val_error,receipt["validation_error"],atol=2e-6,rtol=1e-5):
                            raise AssertionError("candidate validation error mismatch")
                        options.append((receipt["validation_error"],receipt["validation_loss"],index))
                        prediction_checks+=1
                    if min(options)[2]!=row["selected_candidate"]:
                        raise AssertionError("family selection differs from validation ranking")
                rows.append(row)
                if method=="query_aware": effects.extend(records.rows(unit,"interventions"))
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
    families=[]
    for kind in ("classification","regression"):
        for method in METHODS:
            part=[r for r in task_rows if r["task_type"]==kind and r["method"]==method]
            if not part: continue
            baseline=[next(r for r in task_rows if r["task"]==v["task"] and r["method"]=="original") for v in part]
            diff=[v["test_error"]-b["test_error"] for v,b in zip(part,baseline)]
            good=[(v,b) for v,b in zip(part,baseline) if min(v["test_error"],b["test_error"])>0]
            families.append(dict(task_type=kind,method=method,n_tasks=len(part),mean_error=float(np.mean([r["test_error"] for r in part])),
                wins=sum(d < -1e-7 for d in diff),ties=sum(abs(d)<=1e-7 for d in diff),losses=sum(d>1e-7 for d in diff),
                gmr_vs_original=float(np.exp(np.mean([np.log(v["test_error"]/b["test_error"]) for v,b in good]))) if good else None,
                zero_exclusions=[v["task"] for v,b in zip(part,baseline) if min(v["test_error"],b["test_error"])<=0]))
    comparisons=[]
    for kind in ("classification","regression"):
        for reference in ("original","mlp","tabm","catboost","knn"):
            pairs=[]
            for task in tasks:
                if not task.startswith(kind): continue
                selected={r["method"]:r for r in task_rows if r["task"]==task}
                if "query_aware" in selected and reference in selected:
                    pairs.append((selected["query_aware"],selected[reference]))
            if not pairs: continue
            ds=[a["test_error"]-b["test_error"] for a,b in pairs]
            positive=[(a,b) for a,b in pairs if min(a["test_error"],b["test_error"])>0]
            comparisons.append(dict(task_type=kind,reference=reference,n_tasks=len(pairs),
                wins=sum(d < -1e-7 for d in ds),ties=sum(abs(d)<=1e-7 for d in ds),losses=sum(d>1e-7 for d in ds),
                gmr=float(np.exp(np.mean([np.log(a["test_error"]/b["test_error"]) for a,b in positive]))) if positive else None,
                zero_exclusions=[a["task"] for a,b in pairs if min(a["test_error"],b["test_error"])<=0]))
    adaptation=[]
    intervention_summary=[]
    for kind in ("classification","regression"):
        for method in ("original","query_aware"):
            part=[r for r in rows if r["task_type"]==kind and r["method"]==method and "test_helped" in r]
            if part:
                adaptation.append(dict(task_type=kind,method=method,n_runs=len(part),
                    mean_pre_error=float(np.mean([r["test_pre_error"] for r in part])),
                    mean_final_error=float(np.mean([r["test_error"] for r in part])),
                    helped=sum(r["test_helped"] for r in part),harmed=sum(r["test_harmed"] for r in part),
                    n=sum(r["test_n"] for r in part)))
            for intervention in PROTOCOL["interventions"]:
                part=[r for r in effects if r["task_type"]==kind and r["method"]==method and r["intervention"]==intervention]
                if not part: continue
                item=dict(task_type=kind,method=method,intervention=intervention,n_runs=len(part))
                for field in ("error","baseline_error","output_rms_change","retrieval_rms_change","sensitivity_ratio","class_change_rate"):
                    values=[r[field] for r in part if r.get(field) is not None]
                    if values: item["mean_"+field]=float(np.mean(values))
                intervention_summary.append(item)
    report=dict(study=study,protocol=PROTOCOL,baseline_candidates=BASELINES,missing=missing,prediction_checks=prediction_checks,
                rows=rows,tasks=task_rows,families=families,comparisons=comparisons,interventions=effects,
                adaptation=adaptation,intervention_summary=intervention_summary)
    if not write:
        return report
    out=REPO/"results"/"adaptation_improvement"/study
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name,items in (("runs",rows),("tasks",task_rows),("families",families),("comparisons",comparisons),
                       ("interventions",effects),("adaptation",adaptation),("intervention_summary",intervention_summary)):
        if not items: continue
        with (out/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)))
            writer.writeheader(); writer.writerows(items)
    if not missing: plot_summary(report,out)
    print(json.dumps(dict(rows=len(rows),missing=len(missing),prediction_checks=prediction_checks,summary=str(out/"comparison.json"))),flush=True)
    return report


def plot_summary(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names={"retrieval_only":"Pretrained NN-kNN retrieval", "original":"Original full NN-kNN",
           "query_aware":"Query-aware full NN-kNN", "knn":"kNN", "mlp":"MLP", "tabm":"TabM", "catboost":"CatBoost"}
    kinds=[k for k in ("classification","regression") if any(r["task_type"]==k for r in report["families"])]
    fig,axes=plt.subplots(1,len(kinds),figsize=(7*len(kinds),5),squeeze=False,sharey=True,layout="constrained")
    for ax,kind in zip(axes[0],kinds):
        family={r["method"]:r for r in report["families"] if r["task_type"]==kind}
        values=[family[m]["mean_error"]*(100 if kind=="classification" else 1) for m in METHODS]
        colors=["#24797d" if m=="query_aware" else "#637c93" if m=="original" else "#a8b8c6" for m in METHODS]
        ax.barh([names[m] for m in METHODS],values,color=colors)
        ax.set_ylim(len(METHODS)-.5,-.5)
        ax.set_xlim(0,max(values)*1.22)
        ax.set_title(kind.capitalize()+f" / {family['original']['n_tasks']} tasks")
        ax.set_xlabel("Mean classification error (%)" if kind=="classification" else "Mean standardized RMSE")
        for i,value in enumerate(values):
            ax.text(value,i,"  "+(f"{value:.3f}" if kind=="classification" else f"{value:.6f}"),va="center",fontsize=9)
    fig.suptitle("Query-aware residual adaptation and matched local baselines")
    fig.savefig(out/"query_aware_comparison.png",dpi=170)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--mode",choices=("run","data","query","baselines","summarize"),default="run")
    parser.add_argument("--tasks",nargs="+",default=prior.TASKS)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(prior.SEEDS))
    args=parser.parse_args(argv)
    if not args.tasks or len(set(args.tasks))!=len(args.tasks) or not set(args.tasks)<=set(prior.TASKS): parser.error("choose unique local tasks")
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(prior.SEEDS): parser.error("choose unique seeds 40-42")
    if args.study in ("am1","ag1","ag2"): parser.error("use a new study for query-aware runs")
    records.check_study(args.study)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode=="summarize": return summarize(args.study,args.tasks,args.seeds)
    for task in args.tasks: run_task(args.study,task,args.seeds,args.mode)


if __name__=="__main__": main()

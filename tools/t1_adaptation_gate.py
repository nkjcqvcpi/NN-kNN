"""Recorded local apply-or-skip experiments on completed old-adapter runs.

Run: --study ag1 --mode run; summarize: --study ag1 --mode summarize.
Completed units are loaded, never fitted/evaluated again. No retrieval training.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
os.environ.setdefault("NNKNN_DEVICE","cpu")

import numpy as np
import torch

from model.common import tracking
from model.common.core import CoreConfig, build_model
from model.nn_cdh import NNCDHAdapter, ClassificationNNCDHAdapter
from model.t1 import records
from model.t1.adaptation_gate import ApplyOrSkipAdapter, MODES, outcome, train_continuation, losses_per_example
from model.t1.magnitude_direction import neighborhood_statistics
from tools.t1_adaptation_improvement import suite, ComparisonRunner, EXPERIMENT as SOURCE_EXPERIMENT, data_fingerprint
from tools.t1_adaptation_diagnostics import TASKS
from tools.t1_run import Runner

EXPERIMENT="gate_v1"
SEEDS=(40,41,42)
STATIC=("retrieval","original","global_switch","global_shrink","disagreement_gate")
PROTOCOL=dict(version=1, epochs=60, patience=12, batch_size=64, max_lr=.001,
              gate_weight=1., gate_threshold=.5, feature_std_floor=.001, feature_clip=10.,
              shrink_grid=[0.,.25,.5,.75,1.], disagreement_quantiles=[0.,.25,.5,.75,1.],
              selection="val_error_then_ce_for_classification_mse_for_regression",
              retrieval="frozen_old_selected_core", train_queries="stable_id_leave_one_out")


def source_for(task, seed, source_study):
    cfg=suite(tasks=[task])[task]
    probe=object.__new__(ComparisonRunner)
    probe.cfg,probe.study,probe.exp=cfg,source_study,SOURCE_EXPERIMENT
    ds=cfg["datasets"][0]
    label,cond=probe.pair_units(ds)[1][0][1]
    key=probe.unit_key(ds,label,seed)
    unit=tracking.load_unit(key,config=None,arrays=False)
    if unit is None: raise ValueError(f"Missing completed old adapter {key.run_id}")
    fingerprint=unit.config["data_fingerprint"]
    probe.fingerprints={seed:fingerprint}
    expected=probe.unit_config({"experiment":SOURCE_EXPERIMENT,"dataset":ds,"seed":seed,
            "condition":cond,"config":cfg,"data_fingerprint":fingerprint})
    if tracking.config_identity(expected)!=tracking.config_identity(unit.config):
        raise ValueError("source configuration mismatch")
    return cfg,probe,unit,fingerprint


def settings_from(description):
    sync=description.get("sync")
    cfg=sync if sync is not None else description["adapter"]
    if cfg.get("nominal_fields"): raise ValueError("this local prototype requires represented numeric inputs")
    if cfg["output_mode"] not in {"nominal_residual_scores","aggregate_regression"}:
        raise ValueError("gate prototype requires additive score/residual adaptation")
    if sync is None:
        vpost=cfg["lambda_cls"] if cfg["loss"]!="diff_only" else 0.
        vdelta=cfg["lambda_diff"] if cfg["loss"]!="cls_only" else 0.
        if cfg["loss"]=="diff_only": vdelta=1.
        if cfg["loss"]=="cls_only": vpost=1.
        small,fr=0.,None
    else:
        vpost,vdelta,small=cfg["v_post"],cfg["v_delta"],cfg["v_small"]
        fr=description["free_radius_selected"]
    return dict(v_post=vpost,v_delta=vdelta,v_small=small,free_radius=fr,
                score_scale=cfg.get("score_scale",1.), lr=min(cfg.get("adapter_lr",cfg.get("lr",.001)),PROTOCOL["max_lr"]),
                weight_decay=cfg.get("adapter_weight_decay",cfg.get("weight_decay",0.)),
                epochs=PROTOCOL["epochs"],patience=PROTOCOL["patience"],batch_size=PROTOCOL["batch_size"])


def config_for(task,seed,source_study,source_unit,fingerprint,arm):
    return dict(experiment=EXPERIMENT,task=task,task_type=task.split("__")[0],seed=seed,arm=arm,
                source_study=source_study,source_unit=source_unit.info.run_id,
                source_config_identity=tracking.config_identity(source_unit.config),
                data_fingerprint=fingerprint,protocol=PROTOCOL)


def key_for(study,task,seed,arm):
    # Task-family prefix unnecessary for this fixed suite, keeping Windows paths short.
    return records.key(study,EXPERIMENT,task.split("__",1)[1],arm,f"s{seed}")


@torch.no_grad()
def make_inputs(model,data):
    output={}
    for split in ("train","val","test"):
        X,y=getattr(data,"X_"+split),getattr(data,"y_"+split)
        groups=[]
        for start in range(0,len(y),128):
            end=min(start+128,len(y))
            ids=torch.arange(start,end) if split=="train" else None
            r=model.retrieve(X[start:end],query_case_ids=ids,exclude_identical=False)
            w=r["weights"]
            labels=model.labels[r["case_indices"]].float()
            p0=w@labels
            if data.task_type=="classification":
                p0=p0/p0.sum(1,keepdim=True).clamp_min(1e-12)
            dz=r["query_features"]-w@r["case_features"]
            stats=neighborhood_statistics(r,labels,data.task_type)
            if split=="train":
                active_ids=model.active_case_ids()[r["case_indices"]]
                self_mask=ids[:,None]==active_ids[None,:]
                if w.masked_select(self_mask).ne(0).any(): raise AssertionError("self retrieval leaked into training")
            groups.append(dict(x=torch.cat((dz,p0),1),p0=p0,statistics=stats,target=y[start:end]))
        output[split]={k:torch.cat([r[k] for r in groups]).cpu() for k in groups[0]}
    return output


def cache_inputs(study,task,seed,source_study,source,probe,fingerprint,*,config):
    key=key_for(study,task,seed,"inputs")
    saved=tracking.load_unit(key,config=config,arrays=False)
    if saved is not None:
        return saved.load_state("inputs")
    # This evaluation is recorded and never overwrites the source checkpoint.
    with tracking.open_run(key,config=config) as run:
        data=probe.data_for(probe.cfg["datasets"][0],seed)
        if data_fingerprint(data)!=fingerprint: raise ValueError("loaded data differ from source")
        state=source.load_state("checkpoint")
        desc=state["model_description"]
        cc=CoreConfig(**desc["core"])
        # State will overwrite initializer; avoid needless all-pairs initialization.
        cc_load=replace(cc,bias_init="manual")
        labels=state["model_state"]["labels"]
        targets=labels.argmax(1) if data.task_type=="classification" else labels.flatten()
        model=build_model(state["model_state"]["cases"],targets,cc_load,data.num_classes)
        model.load_state_dict(state["model_state"],strict=True)
        model.eval().requires_grad_(False)
        values=make_inputs(model,data)
        options=desc.get("sync",desc.get("adapter"))
        feature_dim=values["train"]["x"].shape[1]-values["train"]["p0"].shape[1]
        if data.task_type=="classification":
            old=ClassificationNNCDHAdapter(feature_dim,data.num_classes,hidden_dims=tuple(options["hidden_dims"]),
                output_mode=options["output_mode"],dropout=options.get("adapter_dropout",options.get("dropout",0.)))
            network=old.net
        else:
            old=NNCDHAdapter(feature_dim,1,tuple(options["hidden_dims"]))
            network=old.adapt_net_final.model
        old.load_state_dict(state["adapter_state"],strict=True)
        settings=settings_from(desc)
        model_gate=ApplyOrSkipAdapter(network,data.task_type,"full_control")
        model_gate.fit_normalizer(values["train"])
        test_result,p=outcome(model_gate,values["test"],settings)
        source_row=records.record(source,"row")
        expected=1-source_row["test_accuracy_post"] if data.task_type=="classification" else source_row["test_rmse_post"]
        if not np.isclose(test_result["error"],expected,atol=2e-6,rtol=1e-5):
            raise AssertionError("reloaded source does not reproduce old final metric")
        events=sorted(records.rows(source,"retrieval_events"),key=lambda e:e["query_index"])
        if len(events)!=len(values["test"]["target"]): raise AssertionError("source event count mismatch")
        expected_outputs=np.array([e["final_values"] if data.task_type=="classification" else [e["final_prediction"]] for e in events])
        if not np.allclose(p.full.numpy(),expected_outputs,atol=2e-6,rtol=1e-5):
            raise AssertionError("source individual predictions not reproduced")
        target=np.array([e["target"] for e in events])
        if not np.array_equal(target,values["test"]["target"].numpy()): raise AssertionError("source targets differ")
        payload=dict(values=values, network=records.pack(network), settings=settings, task_type=data.task_type,
                     source_checkpoint=state, source_unit=source.info.run_id,
                     original_test_error=test_result["error"])
        run.save_state("inputs",records.storable(payload))
        run.log_record("verification",dict(source_predictions_reproduced=True,data_fingerprint=fingerprint,
                    n_train=len(data.y_train),n_val=len(data.y_val),n_test=len(data.y_test)))
        records.save_rng(run)
    return payload


def metric_result(values,scores,apply,kind,settings,full):
    target,p0=values["target"],values["p0"]
    if kind=="classification":
        pre=p0.argmax(1)!=target.long()
        post=scores.argmax(1)!=target.long()
        full_error=full.argmax(1)!=target.long()
        error=float(post.float().mean())
        pre_error=float(pre.float().mean())
        helped=int((pre&~post).sum()); harmed=int((~pre&post).sum())
        oracle=float(torch.minimum(pre,full_error).float().mean())
    else:
        pre=losses_per_example(p0,target,kind)
        post=losses_per_example(scores,target,kind)
        error=float(post.mean().sqrt()); pre_error=float(pre.mean().sqrt())
        helped=int((post<pre-1e-8).sum()); harmed=int((post>pre+1e-8).sum())
        oracle=float(torch.minimum(pre,losses_per_example(full,target,kind)).mean().sqrt())
    result=dict(n=len(target),error=error,pre_error=pre_error,helped=helped,harmed=harmed,
                apply_rate=float(apply.float().mean()),task_loss=float(losses_per_example(scores,target,kind,settings["score_scale"]).mean()),
                oracle_branch_error=oracle)
    if kind=="regression":
        c,r=(scores-p0).flatten(),(target.float().view_as(p0)-p0).flatten()
        result.update(wrong_direction=int(((c*r<0)&(c.abs()>1e-8)&(r.abs()>1e-8)).sum()),
                      harmful_overshoot=int(((c*r>0)&(c.abs()>2*r.abs())&(r.abs()>1e-8)).sum()))
    return result


def static_choice(kind,values,full,settings,train_statistics):
    correction=full-values["p0"]
    candidates=[("scale",a) for a in PROTOCOL["shrink_grid"]]
    thresholds=[float(t) for t in torch.quantile(train_statistics[:,3],torch.tensor(PROTOCOL["disagreement_quantiles"]))]
    choices={"retrieval":[("scale",0.)],"original":[("scale",1.)],"global_switch":[("scale",0.),("scale",1.)],
             "global_shrink":candidates,"disagreement_gate":[("scale",0.),("scale",1.)]+[("disagreement",t) for t in thresholds]}
    selected={}
    for name,options in choices.items():
        scored=[]
        for rule,parameter in options:
            weight=torch.full((len(full),),parameter) if rule=="scale" else (values["statistics"][:,3]>parameter).float()
            score=values["p0"]+weight[:,None]*correction
            result=metric_result(values,score,weight!=0,kind,settings,full)
            # Same primary selection metric; exact task-error ties favor less change.
            primary=result["error"] if kind=="classification" else result["task_loss"]
            scored.append(((primary,float(weight.mean())),rule,parameter))
        _,rule,parameter=min(scored,key=lambda x:x[0])
        selected[name]=dict(rule=rule,parameter=parameter)
    return selected


def record_prediction(run,name,values,scores,apply,full,logit=None):
    data=dict(target=values["target"],retrieval=values["p0"],full=full,final=scores,apply=apply)
    if logit is not None: data["gate_probability"]=logit.sigmoid()
    run.save_state(name,records.storable(data))


def run_static(study,task,seed,cache,config):
    key=key_for(study,task,seed,"controls")
    if tracking.load_unit(key,config=config,arrays=False) is not None:
        print(f"[recorded] {task} s{seed} controls",flush=True); return
    with tracking.open_run(key,config=config) as run:
        kind,settings,values=cache["task_type"],cache["settings"],cache["values"]
        model=ApplyOrSkipAdapter(records.unpack(cache["network"]),kind,"full_control")
        model.fit_normalizer(values["train"])
        _,pv=outcome(model,values["val"],settings)
        chosen=static_choice(kind,values["val"],pv.full,settings,values["train"]["statistics"])
        rows=[]
        for name,choice in chosen.items():
            row=dict(task=task,task_type=kind,seed=seed,method=name,**choice)
            for split in ("val","test"):
                _,p=outcome(model,values[split],settings)
                rule,param=choice["rule"],choice["parameter"]
                w=torch.full((len(p.full),),param) if rule=="scale" else (values[split]["statistics"][:,3]>param).float()
                scores=values[split]["p0"]+w[:,None]*p.proposal
                metrics=metric_result(values[split],scores,w!=0,kind,settings,p.full)
                row.update({split+"_"+k:v for k,v in metrics.items()})
                record_prediction(run,name+"_"+split,values[split],scores,w!=0,p.full)
            rows.append(row)
        run.log_records("rows",[records.ordered(r) for r in rows])
        run.log_record("selected_controls",chosen)
        run.log_final({r["method"]+"_error":r["test_error"] for r in rows})
        run.save_state("source_reference",dict(unit=cache["source_unit"]))
        records.save_rng(run)
    print(f"[done] {task} s{seed} controls",flush=True)


def run_learned(study,task,seed,mode,cache,config):
    key=key_for(study,task,seed,mode)
    saved=tracking.load_unit(key,config=config,arrays=False)
    if saved is not None:
        print(f"[recorded] {task} s{seed} {mode}",flush=True); return
    with tracking.open_run(key,config=config) as run:
        model=ApplyOrSkipAdapter(records.unpack(cache["network"]),cache["task_type"],mode)
        info=train_continuation(model,cache["values"]["train"],cache["values"]["val"],cache["settings"],seed,run=run)
        row=dict(task=task,task_type=cache["task_type"],seed=seed,method=mode,**info)
        for split in ("val","test"):
            result,p=outcome(model,cache["values"][split],cache["settings"])
            row.update({split+"_"+k:v for k,v in result.items()})
            record_prediction(run,split+"_predictions",cache["values"][split],p.final,p.apply,p.full,p.logit)
        run.log_records("rows",[records.ordered(row)])
        run.log_final({k:v for k,v in row.items() if isinstance(v,(int,float))})
        run.log_record("source_reference",dict(unit=cache["source_unit"]))
        records.save_rng(run)
    print(f"[done] {task} s{seed} {mode}: error={row['test_error']:.6f}, apply={row['test_apply_rate']:.3f}, epoch={info['selected_epoch']}",flush=True)


def run_task(study,task,seeds,source_study):
    for seed in seeds:
        cfg,probe,source,fingerprint=source_for(task,seed,source_study)
        configs={arm:config_for(task,seed,source_study,source,fingerprint,arm) for arm in ("inputs","controls",*MODES)}
        records.check_unit_paths(study,[(EXPERIMENT,task.split("__",1)[1],arm,f"s{seed}") for arm in configs])
        # Check all units before any expensive work; identity excludes device/source code.
        complete={arm:tracking.load_unit(key_for(study,task,seed,arm),config=config,arrays=False) is not None for arm,config in configs.items()}
        if all(complete.values()):
            print(f"[recorded] {task} s{seed} all units",flush=True); continue
        cache=cache_inputs(study,task,seed,source_study,source,probe,fingerprint,config=configs["inputs"])
        run_static(study,task,seed,cache,configs["controls"])
        for mode in MODES:
            run_learned(study,task,seed,mode,cache,configs[mode])


def summarize(study,tasks,seeds,source_study):
    rows,missing=[],[]
    for task in tasks:
        for seed in seeds:
            _,_,source,fp=source_for(task,seed,source_study)
            for arm in ("controls",*MODES):
                conf=config_for(task,seed,source_study,source,fp,arm)
                key=key_for(study,task,seed,arm)
                saved=tracking.load_unit(key,config=conf,arrays=False)
                if saved is None: missing.append(key.run_id); continue
                rows.extend(records.rows(saved,"rows"))
    tasks_rows=[]
    for task in tasks:
        for method in (*STATIC,*MODES):
            subset=[r for r in rows if r["task"]==task and r["method"]==method]
            if len(subset)!=len(seeds): continue
            item=dict(task=task,task_type=subset[0]["task_type"],method=method,n_seeds=len(subset))
            keys=set.intersection(*(set(r) for r in subset))
            for k in sorted(keys):
                if k.startswith(("val_","test_")) and all(isinstance(r[k],(int,float)) for r in subset):
                    item[k]=sum(r[k] for r in subset) if k.endswith(("_n","_helped","_harmed","_wrong_direction","_harmful_overshoot")) else float(np.mean([r[k] for r in subset]))
            tasks_rows.append(item)
    families=[]
    for kind in ("classification","regression"):
        for method in (*STATIC,*MODES):
            part=[r for r in tasks_rows if r["task_type"]==kind and r["method"]==method]
            if not part: continue
            base=[next(r for r in tasks_rows if r["task"]==v["task"] and r["method"]=="original") for v in part]
            deltas=[v["test_error"]-b["test_error"] for v,b in zip(part,base)]
            positive=[(v,b) for v,b in zip(part,base) if v["test_error"]>0 and b["test_error"]>0]
            families.append(dict(task_type=kind,method=method,n_tasks=len(part),
                mean_error=float(np.mean([r["test_error"] for r in part])),
                mean_apply_rate=float(np.mean([r["test_apply_rate"] for r in part])),
                wins=sum(x<-1e-7 for x in deltas),ties=sum(abs(x)<=1e-7 for x in deltas),losses=sum(x>1e-7 for x in deltas),
                gmr_vs_original=float(np.exp(np.mean([np.log(v["test_error"]/b["test_error"]) for v,b in positive]))) if positive else None,
                zero_exclusions=[v["task"] for v,b in zip(part,base) if v["test_error"]<=0 or b["test_error"]<=0],
                helped=sum(r["test_helped"] for r in part),harmed=sum(r["test_harmed"] for r in part)))
    report=dict(study=study,source_study=source_study,protocol=PROTOCOL,missing=missing,rows=rows,tasks=tasks_rows,families=families)
    out=REPO/"results"/"adaptation_improvement"/study
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name,items in (("runs",rows),("tasks",tasks_rows),("families",families)):
        if not items: continue
        with (out/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)))
            writer.writeheader();writer.writerows(items)
    if not missing:
        plot_summary(report,out)
    print(json.dumps(dict(rows=len(rows),missing=len(missing),summary=str(out/"comparison.json"))),flush=True)
    return report


def plot_summary(report,out):
    """One static scientific figure, generated from the run-unit summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    tasks=sorted({r["task"] for r in report["tasks"]})
    methods=("global_switch","global_shrink","disagreement_gate",*MODES)
    names=("Global\non/off","Global\nshrink","Disagreement\nthreshold","Extra training\n(no gate)","Probability\ngate","Utility\ngate")
    values=[]
    for task in tasks:
        base=next(r["test_error"] for r in report["tasks"] if r["task"]==task and r["method"]=="original")
        values.append([100*(next(r["test_error"] for r in report["tasks"] if r["task"]==task and r["method"]==m)/base-1) if base>0 else np.nan for m in methods])
    values=np.asarray(values)
    limit=max(1.,float(np.nanmax(np.abs(values))))
    fig,ax=plt.subplots(figsize=(11,8))
    im=ax.imshow(values,cmap="RdYlGn_r",norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect="auto")
    labels=[t.split("__")[1].replace("_"," ").replace("diabets","diabetes")+ (" (C)" if t.startswith("classification") else " (R)") for t in tasks]
    ax.set(xticks=np.arange(len(methods)),xticklabels=names,yticks=np.arange(len(tasks)),yticklabels=labels,
           title="Change in test error versus the original full adapter")
    for i in range(len(tasks)):
        for j in range(len(methods)):
            ax.text(j,i,f"{values[i,j]:+.1f}%",ha="center",va="center",fontsize=9,
                    color="white" if abs(values[i,j])>.65*limit else "black")
    nclass=sum(t.startswith("classification") for t in tasks)
    ax.axhline(nclass-.5,color="white",lw=3)
    fig.colorbar(im,ax=ax,pad=.02,label="Relative error change (%); negative is better")
    fig.text(.02,.025,"Study "+report["study"]+"; three-seed task means. C: classification error, R: standardized RMSE. Validation selects all decisions.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(out/"gate_task_effects.png",dpi=170)
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--source-study",default="am1")
    p.add_argument("--mode",choices=("run","summarize"),default="run")
    p.add_argument("--tasks",nargs="+",default=TASKS)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    args=p.parse_args(argv)
    if not set(args.tasks)<=set(TASKS) or len(set(args.tasks))!=len(args.tasks): p.error("choose unique local tasks")
    if not set(args.seeds)<=set(SEEDS) or len(set(args.seeds))!=len(args.seeds): p.error("choose unique seeds 40-42")
    records.check_study(args.study)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode=="run":
        for task in args.tasks: run_task(args.study,task,args.seeds,args.source_study)
    else:
        return summarize(args.study,args.tasks,args.seeds,args.source_study)


if __name__=="__main__":
    main()

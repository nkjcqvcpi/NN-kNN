"""Recorded frozen-retrieval comparison of non-NN-CDH individual-case learners."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from model.common import tracking
from model.t1 import records
from model.t1.pairwise_adaptation import (
    PairNetwork, pair_training_data, local_ridge_predict, predict_pairs, metrics,
)
from model.t1.pairwise_training import fit_neural, fit_forest
from tools import t1_adapter_dropout as drop

s, q = drop.s, drop.q
EXPERIMENT = "pair_v1"
FAMILIES = ("local_ridge", "pair_trees", "pair_neural")
PROTOCOL = dict(version=1, retrieval="frozen_am1_exact_membership_and_weights",
    features="frozen_representations_train_standardized", random_anchors=8,
    pair_distribution="half_retrieval_half_uniform_nonself_equal_query_mass",
    regression="antisymmetric_pair_difference_then_weighted_anchor_average",
    classification="symmetric_same_class_probability_prior_redistribution",
    family_selection="val_error_then_val_loss_then_candidate_id",
    checkpoint_selection="val_final_loss_then_val_error",
    policy_selection="val_error_then_retrieval_ridge_trees_neural",
    test="selected_candidates_only")
NEURAL = dict(hidden=[64,32], lr=.001, weight_decay=1e-4, grad_clip=5.,
    batch_size=512, epochs=120, patience=20)
CANDIDATES = {
    "local_ridge": {f"r{i}":dict(ridge=v) for i,v in enumerate((.01,.1,1.))},
    "pair_trees": {f"t{i}":dict(n_estimators=64,tree_step=16,max_depth=12,max_features=.7,min_samples_leaf=v)
                   for i,v in enumerate((5,20))},
    "pair_neural": {f"n{i}":dict(**NEURAL,dropout=v) for i,v in enumerate((0.,.1))},
}


def key_for(study, task, seed, unit):
    return records.key(study, EXPERIMENT, task.split("__",1)[1], unit, f"s{seed}")


def numpy_state(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, dict):
        return {k:numpy_state(v) for k,v in value.items()}
    if isinstance(value, list):
        return [numpy_state(v) for v in value]
    return value


def context_config(task, seed, refs):
    return dict(experiment=EXPERIMENT, role="context", protocol=PROTOCOL, task=task, seed=seed,
        core_unit=refs[2].info.run_id, core_config_identity=tracking.config_identity(refs[2].config),
        data_fingerprint=refs[4])


def prepare_context(study, task, seed, refs, sources):
    config = context_config(task, seed, refs)
    key = key_for(study,task,seed,"ctx")
    existing = tracking.load_unit(key, config=config, arrays=False)
    if existing is not None:
        return existing, numpy_state(existing.load_state("context"))
    with tracking.open_run(key,config=config) as run:
        resumed = run.load_resume()
        if resumed is not None:
            context = numpy_state(resumed.state["context"])
        else:
            data = s.n.load_data(task,seed,refs)
            base = records.unpack(refs[2].load_state("objects"))
            model = base["model"]
            model.eval()
            model.requires_grad_(False)
            if not torch.equal(model.cases.cpu(), data.X_train.cpu()):
                raise AssertionError("training cases are not aligned with data")
            if not torch.equal(model.case_ids.cpu(), torch.arange(len(data.y_train))):
                raise AssertionError("expected stable training row case IDs")
            if not model.config.get("pre_topk_mask") or model.config.get("sampling_cases_flag"):
                raise ValueError("this study requires fixed top-k retrieval without case sampling")
            initial_hash = s.state_hash(model.state_dict())
            context = dict(task_type=data.task_type, core_hash=initial_hash, data_fingerprint=refs[4])
            raw_case = None
            for split in ("train","val","test"):
                X = getattr(data,"X_"+split)
                batches = []
                with torch.no_grad():
                    for start in range(0,len(X),128):
                        x = X[start:start+128]
                        ids = torch.arange(start,start+len(x)) if split=="train" else None
                        r = model.retrieve(x,exclude_identical=False,query_case_ids=ids)
                        if raw_case is None:
                            if not torch.equal(r["case_indices"].cpu(),torch.arange(len(data.y_train))):
                                raise AssertionError("case order differs")
                            raw_case = r["case_features"].cpu().numpy()
                            context["case_targets"] = model.labels[r["case_indices"]].float().cpu().numpy()
                        p0 = r["weights"]@model.labels[r["case_indices"]].float()
                        if data.task_type == "classification":
                            p0 = p0/p0.sum(1,keepdim=True).clamp_min(1e-12)
                        w, idx = r["weights"].topk(int(model.config["top_k"]), dim=1)
                        idx = r["case_indices"][idx]
                        if not torch.allclose(w.sum(1),torch.ones(len(x)),atol=1e-6):
                            raise AssertionError("top-k does not retain all retrieval mass")
                        if ids is not None and ((idx==ids[:,None]) & (w>0)).any():
                            raise AssertionError("training self-case not excluded")
                        batches.append(dict(query_z=r["query_features"].cpu().numpy(),
                            indices=idx.cpu().numpy(),weights=w.cpu().numpy(),p0=p0.cpu().numpy()))
                context[split] = {k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
                context[split]["target"] = getattr(data,"y_"+split).cpu().numpy()
            mean, std = raw_case.astype(np.float64).mean(0),raw_case.astype(np.float64).std(0)
            std = np.where(std<1e-6,1.,std)
            context["case_z"] = ((raw_case-mean)/std).astype(np.float32)
            context["feature_mean"], context["feature_std"] = mean,std
            if data.task_type == "classification":
                context["prior"] = context["case_targets"].mean(0).astype(np.float64)
                if not np.array_equal(context["case_targets"].argmax(1),context["train"]["target"]):
                    raise AssertionError("case labels differ from training labels")
            else:
                context["prior"] = None
                if not np.array_equal(context["case_targets"].reshape(-1),context["train"]["target"].reshape(-1)):
                    raise AssertionError("case labels differ from training targets")
            for split in ("train","val","test"):
                context[split]["query_z"] = ((context[split]["query_z"]-mean)/std).astype(np.float32)
                if split != "train":
                    ref = drop.zero_unit(task,seed,"query_aware","sequential",refs,sources).load_state(split+"_predictions")
                    if not np.array_equal(ref["target"].numpy(),context[split]["target"]):
                        raise AssertionError("split targets differ")
                    if not np.array_equal(ref["p0"].numpy(),context[split]["p0"]):
                        raise AssertionError("frozen retrieval predictions differ")
            if initial_hash != s.state_hash(model.state_dict()):
                raise AssertionError("retrieval model changed while making context")
            run.save_resume(dict(context=records.storable(context)),step=1)
        run.save_state("context",records.storable(context))
        run.log_record("context",dict(core_hash=context["core_hash"],data_fingerprint=refs[4],
            n_train=len(context["train"]["target"]),n_val=len(context["val"]["target"]),
            n_test=len(context["test"]["target"]),feature_dim=context["case_z"].shape[1],
            neighbors=context["train"]["indices"].shape[1],positive_weight_self_pairs=0))
        run.log_final(dict(n_train=len(context["train"]["target"]),feature_dim=context["case_z"].shape[1]))
        records.save_rng(run)
    print(f"[context] {task} s{seed}",flush=True)
    return tracking.load_unit(key,config=config,arrays=False), context


def candidate_config(task,seed,family,candidate,context_unit):
    return dict(experiment=EXPERIMENT,protocol=PROTOCOL,task=task,seed=seed,family=family,
        candidate=candidate,settings=CANDIDATES[family][candidate],context_unit=context_unit.info.run_id,
        context_config_identity=tracking.config_identity(context_unit.config))


def predict_model(model, context, part, family, settings):
    return local_ridge_predict(context,part,settings["ridge"]) if family=="local_ridge" else predict_pairs(model,context,part,family=family)


def run_candidate(study,task,seed,family,candidate,context_unit,context,pairs=None):
    config = candidate_config(task,seed,family,candidate,context_unit)
    key = key_for(study,task,seed,candidate)
    existing = tracking.load_unit(key,config=config,arrays=False)
    if existing is not None:
        print(f"[recorded] {task} s{seed} {candidate}",flush=True)
        return existing
    settings = CANDIDATES[family][candidate]
    started = time.perf_counter()
    with tracking.open_run(key,config=config) as run:
        if family=="local_ridge":
            run.load_resume()
            model, info = None,dict(history=[],selected_epoch=0,epochs_run=0)
        else:
            pairs = pairs if pairs is not None else pair_training_data(context,seed=seed,random_anchors=PROTOCOL["random_anchors"])
            fn = fit_neural if family=="pair_neural" else fit_forest
            model,info = fn(context,pairs,settings,seed=seed,run=run)
        prediction = predict_model(model,context,context["val"],family,settings)
        val = metrics(prediction,context["val"]["target"],context["task_type"],context["val"]["p0"])
        row = dict(task=task,task_type=context["task_type"],seed=seed,family=family,candidate=candidate,
            unit=key.run_id,elapsed_seconds=time.perf_counter()-started,
            selected_epoch=info["selected_epoch"],epochs_run=info["epochs_run"],
            **{"val_"+k:v for k,v in val.items()})
        run.save_state("model",records.pack(model))
        run.save_state("val_predictions",records.storable(dict(final=prediction,p0=context["val"]["p0"],target=context["val"]["target"])))
        run.log_record("row",row)
        run.log_record("history",info["history"])
        run.log_final({k:v for k,v in row.items() if isinstance(v,(int,float))})
        records.save_rng(run)
    print(f"[candidate] {task} s{seed} {candidate}: val={val['error']:.6f}",flush=True)
    return tracking.load_unit(key,config=config,arrays=False)


def choose_candidate(rows):
    return min(rows,key=lambda r:(r["val_error"],r["val_task_loss"],r["candidate"]))


def selected_config(task,seed,family,candidates,chosen):
    return dict(experiment=EXPERIMENT,protocol=PROTOCOL,task=task,seed=seed,family=family,role="selected",
        candidates=[dict(unit=u.info.run_id,config_identity=tracking.config_identity(u.config)) for u in candidates],
        selected_candidate=chosen["candidate"],selected_unit=chosen["unit"])


def evaluate_selected(study,task,seed,family,candidates,context):
    rows = [records.record(u,"row") for u in candidates]
    chosen = choose_candidate(rows)
    config = selected_config(task,seed,family,candidates,chosen)
    key = key_for(study,task,seed,"eval_"+family.split("_")[-1])
    existing = tracking.load_unit(key,config=config,arrays=False)
    if existing is not None:
        print(f"[recorded-eval] {task} s{seed} {family}",flush=True)
        return existing
    selected = next(u for u in candidates if u.info.run_id==chosen["unit"])
    with tracking.open_run(key,config=config) as run:
        run.load_resume()
        model = records.unpack(selected.load_state("model"))
        settings = CANDIDATES[family][chosen["candidate"]]
        val = predict_model(model,context,context["val"],family,settings)
        if not np.array_equal(val,selected.load_state("val_predictions")["final"].numpy()):
            raise AssertionError("restored candidate validation predictions differ")
        row = dict(chosen,method=family,adapter=family,scheme="sequential",unit=key.run_id,
            candidate_unit=chosen["unit"],reused=False,initial_core_hash=context["core_hash"],
            selected_core_hash=context["core_hash"])
        final = predict_model(model,context,context["test"],family,settings)
        if not np.array_equal(final,predict_model(model,context,context["test"],family,settings)):
            raise AssertionError("test prediction is not deterministic")
        part = context["test"]
        row.update({"test_"+k:v for k,v in metrics(final,part["target"],context["task_type"],part["p0"]).items()})
        row["test_pre_error"] = metrics(part["p0"],part["target"],context["task_type"])["error"]
        run.save_state("test_predictions",records.storable(dict(final=final,p0=part["p0"],target=part["target"])))
        run.log_record("row",row)
        run.log_record("selection",dict(selected=chosen["candidate"],candidates=rows))
        run.log_final({k:v for k,v in row.items() if isinstance(v,(int,float))})
        records.save_rng(run)
    print(f"[selected] {task} s{seed} {family}: test={row['test_error']:.6f}",flush=True)
    return tracking.load_unit(key,config=config,arrays=False)


def run_split(study,task,seed,families=FAMILIES):
    refs,sources = s.references(task,seed)
    context_unit,context = prepare_context(study,task,seed,refs,sources)
    pairs = None
    for family in families:
        if family!="local_ridge" and pairs is None:
            pairs = pair_training_data(context,seed=seed,random_anchors=PROTOCOL["random_anchors"])
        candidates = [run_candidate(study,task,seed,family,c,context_unit,context,pairs) for c in CANDIDATES[family]]
        evaluate_selected(study,task,seed,family,candidates,context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--mode",choices=("run","summarize"),default="run")
    parser.add_argument("--tasks",nargs="+",choices=q.prior.TASKS,default=list(q.prior.TASKS))
    parser.add_argument("--seeds",nargs="+",type=int,default=list(q.prior.SEEDS))
    parser.add_argument("--families",nargs="+",choices=FAMILIES,default=list(FAMILIES))
    args = parser.parse_args(argv)
    records.check_study(args.study)
    if args.mode=="summarize":
        from tools.t1_pairwise_redesign_summary import summarize
        return summarize(args.study,args.tasks,args.seeds)
    for task in args.tasks:
        for seed in args.seeds:
            run_split(args.study,task,seed,args.families)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

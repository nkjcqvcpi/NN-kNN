"""Resumable recorded fitting of individual-case neural and tree learners."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F

from . import records
from .pairwise_adaptation import PairNetwork, pair_features, forest_pair_predict, predict_pairs, metrics


def fit_neural(context, pairs, config, *, seed, run):
    torch.manual_seed(seed)
    model = PairNetwork(context["case_z"].shape[1], context["task_type"],
        dropout=config["dropout"], hidden=tuple(config["hidden"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed+123)
    a, b, target, weight = (torch.from_numpy(pairs[k]) for k in ("a", "b", "target", "weight"))
    best, history, bad, start, stopped = None, [], 0, 1, False
    resumed = run.load_resume()
    if resumed is not None:
        state = resumed.state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, history, bad = state["best"], state["history"], state["bad"]
        start, stopped = state["epoch"]+1, state["stopped"]
        resumed.restore_rng({"shuffle":generator})
    for epoch in range(start, config["epochs"]+1) if not stopped else ():
        model.train()
        order = torch.randperm(len(a), generator=generator)
        loss_total = 0.
        for indices in order.split(config["batch_size"]):
            output = model(a[indices], b[indices])
            if context["task_type"] == "classification":
                losses = F.binary_cross_entropy(output.clamp(1e-7, 1-1e-7), target[indices], reduction="none")
            else:
                losses = (output-target[indices]).square()
            loss = (losses.mean(1)*weight[indices]).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite pair training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            loss_total += float(loss.detach())*len(indices)
        prediction = predict_pairs(model, context, context["val"], family="pair_neural")
        val = metrics(prediction, context["val"]["target"], context["task_type"])
        rec = dict(epoch=epoch, train_pair_loss=loss_total/len(a), val_loss=val["task_loss"], val_error=val["error"])
        history.append(rec)
        if best is None or (rec["val_loss"], rec["val_error"]) < (best["loss"], best["error"]):
            best = dict(epoch=epoch, loss=rec["val_loss"], error=rec["val_error"], model=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
        stopped = bad >= config["patience"]
        run.log_step(epoch, rec)
        run.save_resume(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best=best,
            history=history, bad=bad, epoch=epoch, stopped=stopped), step=epoch, generators={"shuffle":generator})
        if stopped:
            break
    if best is None:
        raise ValueError("neural fitting produced no checkpoint")
    model.load_state_dict(best["model"])
    model.eval()
    return model, dict(history=history, selected_epoch=best["epoch"], epochs_run=len(history))


def fit_forest(context, pairs, config, *, seed, run):
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,
    )
    kind = context["task_type"]
    a, b, y, w = (pairs[k] for k in ("a", "b", "target", "weight"))
    x = np.concatenate((pair_features(a,b), pair_features(b,a)))
    target = np.concatenate((y, y if kind == "classification" else -y)).reshape(-1)
    if kind == "classification":
        target = target.astype(int)
    weight = np.concatenate((w/2, w/2))
    forest_kind = config.get("forest_kind", "extra_trees")
    constructors = {
        "extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor),
        "random_forest": (RandomForestClassifier, RandomForestRegressor),
    }
    if forest_kind not in constructors:
        raise ValueError(f"unknown forest kind: {forest_kind}")
    constructor = constructors[forest_kind][0 if kind == "classification" else 1]
    model = constructor(n_estimators=0, warm_start=True, random_state=seed, n_jobs=1,
        max_depth=config["max_depth"], max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"],
        bootstrap=config.get("bootstrap", forest_kind == "random_forest"))
    history = []
    resumed = run.load_resume()
    if resumed is not None:
        model = records.unpack(resumed.state["model"])
        if type(model) is not constructor or model.bootstrap != config.get("bootstrap", forest_kind == "random_forest"):
            raise ValueError("resumed forest type/bootstrap does not match configuration")
        history = resumed.state["history"]
        resumed.restore_rng()
    for count in range(model.n_estimators+config["tree_step"], config["n_estimators"]+1, config["tree_step"]):
        model.set_params(n_estimators=count)
        model.fit(x, target, sample_weight=weight)
        val_prediction = predict_pairs(model, context, context["val"], family="pair_trees")
        val = metrics(val_prediction, context["val"]["target"], kind)
        # A fixed bounded training probe supplies progress diagnostics.
        take = slice(0,min(4096,len(a)))
        probe = forest_pair_predict(model, a[take], b[take], kind)
        yt = y[take].reshape(-1)
        if kind == "classification":
            p = np.clip(probe,1e-12,1-1e-12)
            losses = -(yt*np.log(p)+(1-yt)*np.log(1-p))
        else:
            losses = np.square(probe-yt)
        rec = dict(trees=count, train_probe_loss=float(np.average(losses,weights=w[take])),
            val_loss=val["task_loss"], val_error=val["error"])
        history.append(rec)
        run.log_step(count, rec)
        run.save_resume(dict(model=records.pack(model), history=history), step=count)
    if model.n_estimators != config["n_estimators"]:
        raise ValueError("incomplete forest")
    return model, dict(history=history, selected_epoch=config["n_estimators"], epochs_run=len(history))

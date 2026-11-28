"""Mandatory recorded, resumable fitting for non-NN-CDH alternatives."""
from __future__ import annotations

import copy
import numpy as np
import torch
from . import records
from .pairwise_adaptation import metrics
from .alternative_adaptation import VoteAttention, attention_tensors, attention_predict, point_predict


def make_folds(context, seed, n_splits=5):
    from sklearn.model_selection import KFold, StratifiedKFold
    z = context["case_z"]
    if context["task_type"] == "classification":
        y = context["case_targets"].argmax(1)
        if np.bincount(y).min() < n_splits:
            raise ValueError("not enough cases per class for declared stratified folds")
        iterator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(z, y)
    else:
        iterator = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(z)
    splits = [(train, held) for train, held in iterator]
    fold_of_case = np.full(len(z), -1, dtype=int)
    for fold, (train, held) in enumerate(splits):
        if np.intersect1d(train, held).size:
            raise AssertionError("cross-fit anchor leaked into its predictor")
        fold_of_case[held] = fold
    if (fold_of_case < 0).any():
        raise AssertionError("incomplete out-of-fold assignment")
    return splits, fold_of_case


def fit_point_forests(context, config, *, seed, run, transport=False):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    kind = context["task_type"]
    z, target = context["case_z"], context["case_targets"]
    y = target.argmax(1) if kind == "classification" else target.reshape(-1)
    n = len(z)
    if transport:
        splits, fold_of_case = make_folds(context, seed, config["folds"])
    else:
        splits, fold_of_case = [(np.arange(n), np.arange(n))], None
    cls = ExtraTreesClassifier if kind == "classification" else ExtraTreesRegressor
    models = [cls(n_estimators=0, warm_start=True, random_state=seed + fold, n_jobs=1,
        max_depth=config["max_depth"], max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"])
        for fold in range(len(splits))]
    history = []
    resumed = run.load_resume()
    if resumed is not None:
        models, history = records.unpack(resumed.state["models"]), resumed.state["history"]
        resumed.restore_rng()
    for fold, ((train, held), model) in enumerate(zip(splits, models)):
        for count in range(model.n_estimators + config["tree_step"], config["n_estimators"] + 1, config["tree_step"]):
            model.set_params(n_estimators=count)
            model.fit(z[train], y[train])
            if transport:
                pred = point_predict(model, z[held], kind, target.shape[1])
                measured = metrics(pred, y[held], kind)
                rec = dict(fold=fold, trees=count, heldout_train_loss=measured["task_loss"], heldout_train_error=measured["error"])
            else:
                part = context["val"]
                measured = metrics(point_predict(model, part["query_z"], kind, target.shape[1]), part["target"], kind)
                rec = dict(fold=fold, trees=count, val_loss=measured["task_loss"], val_error=measured["error"])
            history.append(rec)
            step = fold * config["n_estimators"] + count
            run.log_step(step, rec)
            run.save_resume(dict(models=records.pack(models), history=history), step=step)
        if model.n_estimators != config["n_estimators"]:
            raise AssertionError("incomplete forest")
    if transport:
        anchor = np.zeros_like(target, dtype=np.float64)
        for (_, held), model in zip(splits, models):
            anchor[held] = point_predict(model, z[held], kind, target.shape[1])
        result = dict(models=models, fold_of_case=fold_of_case, anchor_predictions=anchor,
            training_indices=[train for train, _ in splits])
    else:
        result = models[0]
    return result, dict(history=history, selected_epoch=len(models) * config["n_estimators"], epochs_run=len(history))


def fit_attention(context, config, *, seed, run):
    torch.manual_seed(seed)
    model = VoteAttention(context["case_z"].shape[1], context["case_targets"].shape[1],
        hidden=tuple(config["hidden"]), dropout=config["dropout"], score_cap=config["score_cap"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 123)
    tensors = attention_tensors(context, context["train"])
    target = torch.as_tensor(context["train"]["target"])
    best, history, bad, start, stopped = None, [], 0, 1, False
    resumed = run.load_resume()
    if resumed is not None:
        state = resumed.state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, history, bad = state["best"], state["history"], state["bad"]
        start, stopped = state["epoch"] + 1, state["stopped"]
        resumed.restore_rng({"shuffle": generator})
    else:
        value = metrics(attention_predict(model, context, context["val"]), context["val"]["target"], context["task_type"])
        best = dict(epoch=0, loss=value["task_loss"], error=value["error"], model=copy.deepcopy(model.state_dict()))
        history.append(dict(epoch=0, val_loss=value["task_loss"], val_error=value["error"]))
        run.log_step(0, history[-1])
    for epoch in range(start, config["epochs"] + 1) if not stopped else ():
        model.train()
        total = 0.
        order = torch.randperm(len(target), generator=generator)
        for idx in order.split(config["batch_size"]):
            prediction = model(*(t[idx] for t in tensors))
            if context["task_type"] == "classification":
                loss = -prediction[torch.arange(len(idx)), target[idx].long()].clamp_min(1e-12).log().mean()
            else:
                loss = (prediction - target[idx].reshape_as(prediction)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite attention loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        value = metrics(attention_predict(model, context, context["val"]), context["val"]["target"], context["task_type"])
        rec = dict(epoch=epoch, train_loss=total / len(target), val_loss=value["task_loss"], val_error=value["error"])
        history.append(rec)
        if (rec["val_loss"], rec["val_error"]) < (best["loss"], best["error"]):
            best = dict(epoch=epoch, loss=rec["val_loss"], error=rec["val_error"], model=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
        stopped = bad >= config["patience"]
        run.log_step(epoch, rec)
        run.save_resume(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best=best,
            history=history, bad=bad, epoch=epoch, stopped=stopped), step=epoch, generators={"shuffle": generator})
        if stopped:
            break
    model.load_state_dict(best["model"])
    model.eval()
    return model, dict(history=history, selected_epoch=best["epoch"], epochs_run=len(history) - 1)

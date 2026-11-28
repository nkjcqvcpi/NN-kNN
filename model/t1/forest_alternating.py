"""Retrieval updates through detached forest proposals, followed by refitting.

Hard forest decisions and top-k indices are not differentiated. Retrieval
activations retain their graph; no query target enters a prediction function.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .pairwise_adaptation import forest_pair_predict


@torch.no_grad()
def extract_context(model, data, *, splits=("train", "val"), normalizer=None, batch_size=128):
    model.eval()
    model._invalidate_case_cache()
    if model.momentum_encoder is not None or model.config.get("sampling_cases_flag"):
        raise ValueError("alternating forest study requires unsampled retrieval without EMA")
    if not model.config.get("pre_topk_mask") or not model.normalize_over_cases:
        raise ValueError("alternating forest study requires normalized pre-top-k retrieval")
    if not torch.equal(model.case_ids.cpu(), torch.arange(len(data.y_train))):
        raise ValueError("case IDs must remain aligned with training rows")
    if not torch.equal(model.cases.cpu(), data.X_train.cpu()):
        raise ValueError("training cases changed")
    context = dict(task_type=data.task_type)
    raw_case = None
    for split in splits:
        x = getattr(data, "X_" + split).to(model.cases.device)
        batches = []
        for start in range(0, len(x), batch_size):
            query = x[start:start + batch_size]
            ids = torch.arange(start, start + len(query), device=x.device) if split == "train" else None
            r = model.retrieve(query, exclude_identical=False, query_case_ids=ids)
            if not torch.equal(r["case_indices"].cpu(), torch.arange(len(data.y_train))):
                raise ValueError("case order changed")
            if raw_case is None:
                raw_case = r["case_features"].cpu().numpy()
                context["case_targets"] = model.labels[r["case_indices"]].float().cpu().numpy()
            p0 = r["weights"] @ model.labels[r["case_indices"]].float()
            if data.task_type == "classification":
                p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
            w, indices = r["weights"].topk(int(model.config["top_k"]), dim=1)
            indices = r["case_indices"][indices]
            if not torch.allclose(w.sum(1), torch.ones(len(query), device=x.device), atol=1e-6):
                raise ValueError("top-k lost retrieval mass")
            if ids is not None and ((indices == ids[:, None]) & (w > 0)).any():
                raise ValueError("positive-weight self retrieval")
            batches.append(dict(query_z=r["query_features"].cpu().numpy(), indices=indices.cpu().numpy(),
                weights=w.cpu().numpy(), p0=p0.cpu().numpy()))
        context[split] = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        context[split]["target"] = getattr(data, "y_" + split).cpu().numpy()
    if normalizer is None:
        mean, std = raw_case.astype(np.float64).mean(0), raw_case.astype(np.float64).std(0)
        std = np.where(std < 1e-6, 1., std)
    else:
        mean, std = normalizer["feature_mean"], normalizer["feature_std"]
    context.update(feature_mean=mean, feature_std=std, case_z=((raw_case - mean) / std).astype(np.float32),
        prior=context["case_targets"].mean(0).astype(np.float64) if data.task_type == "classification" else None)
    for split in splits:
        context[split]["query_z"] = ((context[split]["query_z"] - mean) / std).astype(np.float32)
    model._invalidate_case_cache()
    return context


def detached_proposals(forest, query_features, case_features, labels, indices, context):
    """Return fixed per-case distributions or residuals in float64."""
    mean, std = context["feature_mean"], context["feature_std"]
    query = ((query_features.detach().cpu().numpy() - mean) / std).astype(np.float32)
    cases = ((case_features.detach().cpu().numpy() - mean) / std).astype(np.float32)
    idx = indices.detach().cpu().numpy()
    a = np.broadcast_to(query[:, None, :], (*idx.shape, query.shape[-1])).reshape(-1, query.shape[-1]).copy()
    b = cases[idx].reshape(a.shape)
    values = forest_pair_predict(forest, a, b, context["task_type"]).reshape(idx.shape)
    if context["task_type"] == "classification":
        targets = labels.detach().cpu().numpy()[idx].astype(np.float64)
        prior = context["prior"]
        anchor_prior = (targets * prior).sum(-1)
        proposals = (1 - values)[..., None] * prior / (1 - anchor_prior)[..., None]
        proposals = proposals * (1 - targets) + values[..., None] * targets
    else:
        proposals = values
    return torch.from_numpy(np.ascontiguousarray(proposals)).to(query_features.device)


def online_prediction(model, query, forest, context, *, query_case_ids=None):
    r = model.retrieve(query, exclude_identical=False, query_case_ids=query_case_ids)
    labels = model.labels[r["case_indices"]].float()
    p0 = r["weights"] @ labels
    if context["task_type"] == "classification":
        p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
    weights, indices = r["weights"].topk(int(model.config["top_k"]), dim=1)
    proposals = detached_proposals(forest, r["query_features"], r["case_features"], labels, indices, context)
    if proposals.requires_grad:
        raise AssertionError("forest output unexpectedly retained a gradient")
    if context["task_type"] == "classification":
        final = (weights[..., None] * proposals).sum(1)
        final = final / final.sum(1, keepdim=True).clamp_min(1e-12)
    else:
        final = p0 + (weights * proposals).sum(1, keepdim=True)
    return p0, final


def task_loss(prediction, target, task_type):
    if task_type == "classification":
        return F.nll_loss(prediction.clamp_min(1e-12).log(), target.long())
    return F.mse_loss(prediction, target.reshape_as(prediction).to(prediction))


def train_retrieval_block(model, optimizer, data, forest, context, core_config, *, epochs, seed, run):
    """Continue one retrieval block; every completed epoch is resumable."""
    if run is None or epochs < 1:
        raise ValueError("a recorded run and positive epoch budget are required")
    generator = torch.Generator().manual_seed(seed)
    history, first = [], 1
    resumed = run.load_resume()
    if resumed is not None:
        state = resumed.state
        model.load_state_dict(state["model"])
        model._invalidate_case_cache()
        optimizer.load_state_dict(state["optimizer"])
        history, first = state["history"], state["epoch"] + 1
        resumed.restore_rng({"shuffle": generator})
    x, y = data.X_train.to(model.cases.device), data.y_train.to(model.cases.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(first, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        pre_sum = post_sum = grad_sum = 0.
        updates, post_grad = 0, 0.
        for batch in order.split(core_config.batch_size):
            batch = batch.to(x.device)
            p0, final = online_prediction(model, x[batch], forest, context, query_case_ids=batch)
            pre, post = task_loss(p0, y[batch], data.task_type), task_loss(final, y[batch], data.task_type)
            loss = pre + post
            if core_config.lambda_case_bias:
                loss = loss + core_config.lambda_case_bias * model.biases.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite retrieval objective")
            if updates == 0:
                grads = torch.autograd.grad(post, parameters, retain_graph=True, allow_unused=True)
                post_grad = float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, core_config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            pre_sum += float(pre.detach()) * len(batch)
            post_sum += float(post.detach()) * len(batch)
            grad_sum += float(grad_norm)
            updates += 1
        rec = dict(epoch=epoch, train_pre_loss=pre_sum / len(x), train_final_loss=post_sum / len(x),
            gradient_norm=grad_sum / updates, post_gradient_norm_first_batch=post_grad, retrieval_updates=updates)
        history.append(rec)
        run.log_step(epoch, rec)
        run.save_resume(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
            history=history), step=epoch, generators={"shuffle": generator})
    model.eval()
    model._invalidate_case_cache()
    return history


def neighborhood_drift(before, after):
    """Positive-weight membership and activation changes on aligned queries."""
    changes, l1, overlap = [], [], []
    for old_i, old_w, new_i, new_w in zip(before["indices"], before["weights"], after["indices"], after["weights"]):
        old = {int(i): float(w) for i, w in zip(old_i, old_w) if w > 0}
        new = {int(i): float(w) for i, w in zip(new_i, new_w) if w > 0}
        union = old.keys() | new.keys()
        changes.append(old.keys() != new.keys())
        overlap.append(len(old.keys() & new.keys()) / len(union))
        l1.append(sum(abs(old.get(i, 0.) - new.get(i, 0.)) for i in union))
    return dict(neighborhood_changed_fraction=float(np.mean(changes)),
        mean_neighborhood_jaccard=float(np.mean(overlap)), mean_activation_l1=float(np.mean(l1)))

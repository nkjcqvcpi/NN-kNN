"""Summary of handoff pack p0a (2026-11-26): retention with ``statistics.bias_source: delta_from_init`` and the
build-time initial bias (study handoff_deltainit_r760) against the recorded current-bias references.
Exploratory, our choices, 3 seeds (30-32).

    # 1. on the host whose run root holds the units (r760: source ../env.sh first); reads ONLY runs/
    python tools/t1_handoff1126_p0a.py dump --out p0a_pairs.jsonl [--workers 8]
    # 2. anywhere
    python tools/t1_handoff1126_p0a.py summarize p0a_pairs.jsonl --out results/t1/handoff_20261126/p0a

Pairs (same dataset, K and seed):
- ``deltainit_percentile__<type>`` (within_cohort_percentile) with ``plan_full_r760/retention__<type>`` bias_normalized;
- ``deltainit_medianmad__<type>`` (median_mad) with ``handoff_btransform_r760/bnorm_medianmad__<type>`` bias_normalized.

Per pair ``dump`` records: both statuses and hosts, the recorded ``statistics`` block of the candidate, whether the
per-case maintenance actions (keep / archive per case id) are identical, the largest |policy_score| difference, the
largest |difference| over the numeric ``final/`` scalars, and whether every tensor of the final checkpoint's
``model_state`` and ``adapter_state`` is bitwise equal (the stored ``case_statistics`` differ by construction: they hold
the initial bias). Read-only; no default or other tool changes.

Note (found with this summary): a bias that training drove to a denormal-size value (|b| ~ 1e-38) gives
b - b_build == -b_build in float64, so such cases become exact ties under delta_from_init although their current biases
differ. Where that happens (abalone) the within-cohort percentile B differs from the current-bias B.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEEDS = ("s30", "s31", "s32")
KS = ("0.2", "0.5", "0.75")
VARIANTS = {"percentile": ("plan_full_r760", "retention"), "medianmad": ("handoff_btransform_r760", "bnorm_medianmad")}


def _flat(t, p=""):
    if isinstance(t, dict):
        for k, v in t.items():
            yield from _flat(v, f"{p}/{k}")
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t):
            yield from _flat(v, f"{p}/{i}")
    else:
        yield p, t


def _unit(path: str) -> dict:
    import torch
    from model.common.tracking import TrackingError, load_run, run_status
    from model.t1.records import unordered
    try:
        d = load_run(path, arrays=False)
    except TrackingError:
        return {"status": run_status(path).status}
    ev = sorted(p.name for p in Path(path).iterdir() if "tfevents" in p.name)
    acts = {}
    score = {}
    for r in d.records("case_maintenance"):
        r = unordered(r)
        acts[int(r["case_id"])] = r.get("action")
        score[int(r["case_id"])] = r.get("policy_score")
    ck = d.load_state("checkpoint")
    tens = {k: v for k, v in _flat({"model_state": ck.get("model_state"), "adapter_state": ck.get("adapter_state")})}
    fin = {k: v for k, v in d.final().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"status": "complete", "host": ev[-1].split(".")[4] if ev else "", "stats": ((d.config or {}).get("config") or {}).get("statistics"),
            "acts": acts, "score": score, "tens": tens, "final": fin}


def _pair(job: tuple) -> str:
    import math
    import torch
    rec, cand, ref = job
    a, b = _unit(cand), _unit(ref)
    out = {**rec, "cand_status": a["status"], "ref_status": b["status"]}
    if a["status"] == b["status"] == "complete":
        same_t = set(a["tens"]) == set(b["tens"])
        if same_t:
            for k, x in a["tens"].items():
                y = b["tens"][k]
                if torch.is_tensor(x):
                    if not (torch.is_tensor(y) and x.shape == y.shape and torch.equal(x, y)):
                        same_t = False
                        break
                elif x != y:
                    same_t = False
                    break
        ds = [abs(a["score"][i] - b["score"].get(i, math.nan)) for i in a["score"]
              if a["score"][i] is not None and b["score"].get(i) is not None]
        keys = sorted(set(a["final"]) & set(b["final"]))
        fd = {k: a["final"][k] - b["final"][k] for k in keys}
        out.update(cand_host=a["host"], ref_host=b["host"], stats=a["stats"],
                   same_actions=a["acts"] == b["acts"], n_cases=len(a["acts"]),
                   n_kept=sum(1 for v in a["acts"].values() if v != "archive"),
                   max_score_diff=max(ds) if ds else None, same_model_state=same_t,
                   max_final_diff=max((abs(v) for v in fd.values()), default=None),
                   final_cand=a["final"], final_ref=b["final"])
    return json.dumps(out, default=str)


def dump(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.common.tracking import runs_root
    t1 = runs_root() / "t1"
    jobs = []
    for var, (ref_study, ref_prefix) in VARIANTS.items():
        for tt in ("classification", "regression"):
            ce = t1 / "handoff_deltainit_r760" / f"deltainit_{var}__{tt}"
            re_ = t1 / ref_study / f"{ref_prefix}__{tt}"
            if not ce.is_dir():
                continue
            for dd in sorted(p for p in ce.iterdir() if p.is_dir()):
                for k in KS:
                    for s in SEEDS:
                        run = f"bias_normalized_K{k}"
                        jobs.append(({"variant": var, "task_type": tt, "dataset": dd.name, "K": k, "seed": s},
                                     str(dd / run / s), str(re_ / dd.name / run / s)))
    with open(args.out, "w") as fh, ProcessPoolExecutor(args.workers) as ex:
        for n, line in enumerate(ex.map(_pair, jobs, chunksize=4), 1):
            fh.write(line + "\n")
            if n % 50 == 0:
                print(n, "/", len(jobs), file=sys.stderr, flush=True)
    print(len(jobs), "pairs ->", args.out, file=sys.stderr)


def summarize(args: argparse.Namespace) -> None:
    import pandas as pd
    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    keep = ["variant", "task_type", "dataset", "K", "seed", "cand_status", "ref_status", "cand_host", "ref_host",
            "same_actions", "same_model_state", "max_score_diff", "max_final_diff", "n_cases", "n_kept"]
    df[[c for c in keep if c in df]].to_csv(out / "pairs.csv", index=False)
    ok = df[(df.cand_status == "complete") & (df.ref_status == "complete")]
    df["initial_bias"] = df["stats"].map(lambda s: (s or {}).get("initial_bias") if isinstance(s, dict) else None)
    df["bias_source"] = df["stats"].map(lambda s: (s or {}).get("bias_source") if isinstance(s, dict) else None)
    g = df.groupby(["variant", "task_type"])
    summ = pd.DataFrame({
        "datasets": g.dataset.nunique(),
        "pairs": g.size(),
        "both_complete": g.apply(lambda x: int(((x.cand_status == "complete") & (x.ref_status == "complete")).sum())),
        "same_host": g.apply(lambda x: int((x.cand_host == x.ref_host).sum())),
        "build_time_delta_from_init": g.apply(lambda x: int(((x.initial_bias == "build_time")
                                                             & (x.bias_source == "delta_from_init")).sum())),
        "same_actions": g.apply(lambda x: int(x.same_actions.fillna(False).astype(bool).sum())),
        "same_model_state": g.apply(lambda x: int(x.same_model_state.fillna(False).astype(bool).sum())),
        "max_score_diff": g.max_score_diff.max(),
        "max_final_diff": g.max_final_diff.max(),
    })
    summ.to_csv(out / "summary.csv")
    lines = ["# Pack p0a: delta_from_init (build-time initial bias) vs current-bias references", "",
             "| variant | type | datasets | pairs | both complete | same host | build_time + delta_from_init | "
             "same keep/archive actions | same final model state | max abs policy_score diff | max abs final-scalar diff |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (v, t), r in summ.iterrows():
        lines.append(f"| {v} | {t} | {int(r.datasets)} | {int(r.pairs)} | {int(r.both_complete)} | {int(r.same_host)} | "
                     f"{int(r.build_time_delta_from_init)} | {int(r.same_actions)} | {int(r.same_model_state)} | "
                     f"{r.max_score_diff:.3g} | {r.max_final_diff:.3g} |")
    bad = ok[~(ok.same_actions.astype(bool) & ok.same_model_state.astype(bool))]
    lines += ["", f"Pairs with any difference in actions or model state: {len(bad)}"]
    for _, r in bad.iterrows():
        lines.append(f"- {r.variant} {r.task_type} {r.dataset} K{r.K} {r.seed}: actions {r.same_actions}, "
                     f"model {r.same_model_state}, max final diff {r.max_final_diff}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("summarize")
    s.add_argument("dump")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else summarize(a)


if __name__ == "__main__":
    main()

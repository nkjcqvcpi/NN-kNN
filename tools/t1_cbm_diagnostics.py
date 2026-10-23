"""Read-only case-base maintenance (CBM) diagnostics from saved ``tools/t1_run.py`` outputs.

    python tools/t1_cbm_diagnostics.py <runs dir> [--K 0.5] [--condition GLOB ...]
        [--seeds 20 21 ... | --all-seeds] [--draws 200] [--tie-seed 0] [--json OUT] [--md OUT]

Every ``<task>/<condition>/s<seed>/`` under ``<runs dir>`` with a nonempty case_statistics.jsonl
and a run_manifest.json is read; nothing is trained, no model or checkpoint is loaded and nothing
under ``<runs dir>`` is written. The scores are rebuilt from the run's final train-LOO audit with
``model.t1.provenance.score_cases`` and its statistics settings, over the active cases, as
``maintenance.run_maintenance`` scores them. Per run, at capacity K (``t1_run.resolve_K``):

- exposure (review F4): share never retrieved (R = 0), share evidenced, share of the activation
  mass A held by the top 10% of cases;
- Q as an exposure measure (F1): Spearman(Q, A) and the share with H = 0, among evidenced cases;
- ``tie_decided_share`` (F3): of the cases a provenance_only selection archives, the share whose
  policy score is the tie value 0.5; ``cut_tie_share`` per policy is the exact count, the archived
  cases an equal-ranked kept case competed with for the same slot;
- kept sets of provenance_only (Q), bias_current (B) and an exposure-only control (rank by A,
  never-retrieved first out) from ``retention.select_active_set`` with the run's protections and
  cohort floors (condition keys override the base, as ``t1_run.retention_cfg_for``); Jaccard over
  all cases and over R > 0 cases; share of A each policy archives;
- per-cohort kept fractions under the case-ID tie-break vs ``--draws`` seeded random tie-breaks.
  The draws use a replica of select_active_set's fill that must reproduce its case-ID result.

Logged decisions in case_maintenance.jsonl are replayed through select_active_set from their own
logged scores, a check that the offline selection is the runner's. The markdown has one table per
task, the mean over the selected seeds (default: development seeds 20-25).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import warnings
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.t1.provenance import CaseScores, CaseStatisticsStore, CaseStats, ScoreConfig, score_cases  # noqa: E402
from model.t1.retention import RetentionConfig, candidate_rank, protection_requirements, select_active_set  # noqa: E402

POLICIES = ("provenance_only", "bias_current", "exposure_only")
PAIRS = (("provenance_only", "bias_current"), ("provenance_only", "exposure_only"), ("bias_current", "exposure_only"))
TIE = 0.5  # provenance_only score of an unevidenced case (retention.candidate_rank)
DEV_SEEDS = tuple(range(20, 26))
# Fill policies whose ranking a logged maintenance row fully determines (no distances, no extra scores).
REPLAYABLE = {"full_memory", "random", "stratified", "bias_current", "bias_normalized", "provenance_only"}


def resolve_K(k, n: int) -> int:
    """``tools/t1_run.resolve_K``: a float <= 1 is a fraction of the active cases."""
    if isinstance(k, float) and k <= 1.0:
        return max(1, int(round(k * n)))
    return int(k)


def parse_K(text: str):
    return float(text) if any(ch in text for ch in ".eE") else int(text)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_config(manifest: dict) -> tuple[ScoreConfig, RetentionConfig]:
    """Score settings (``t1_run.score_cfg_for``) and retention settings (``retention_cfg_for``)."""
    rc = manifest["resolved_config"]
    s = rc["config"]["statistics"]
    scfg = ScoreConfig(smoothing=float(s["smoothing"]), min_retrieval_count=float(s["min_retrieval_count"]),
                       min_activation_mass=float(s["min_activation_mass"]),
                       bias_normalization=s.get("bias_normalization", "within_cohort_percentile"),
                       bias_source=s.get("bias_source", "current"))
    known = RetentionConfig.__dataclass_fields__
    r = dict(rc["config"]["retention"])
    r.update({k: v for k, v in rc.get("condition", {}).items() if k in known})
    r["seed"] = int(manifest["run"]["seed"])
    r.setdefault("notes", {})
    return scfg, RetentionConfig(**{k: v for k, v in r.items() if k in known})


def load_scores(rows: list[dict], scfg: ScoreConfig) -> tuple[CaseScores, int]:
    """CaseScores of the active cases and the number of archived rows."""
    active = [r for r in rows if r.get("archive_state", "active") == "active"]
    keys = {f.name for f in fields(CaseStats)}
    store = CaseStatisticsStore.from_state_dict({
        "case_role": active[0].get("case_role", "classification") if active else "classification",
        "stats": {int(r["case_id"]): {k: v for k, v in r.items() if k in keys} for r in active}})
    ids = np.array([int(r["case_id"]) for r in active], dtype=np.int64)
    biases = np.array([float(r["current_bias"]) for r in active])
    return score_cases(ids, [str(r["cohort"]) for r in active], biases, store, scfg), len(rows) - len(active)


def _cfg(rcfg: RetentionConfig, policy: str) -> RetentionConfig:
    """Selection settings of ``policy`` with the case-ID tie-break (the frozen default)."""
    over = {"tie_break": "case_id"} if "tie_break" in RetentionConfig.__dataclass_fields__ else {}
    return replace(rcfg, policy="provenance_only" if policy == "exposure_only" else policy, **over)


def ranking(scores: CaseScores, policy: str, rcfg: RetentionConfig) -> np.ndarray:
    if policy == "exposure_only":
        return np.where(scores.R > 0, scores.A, -1.0)  # A >= 0: never-retrieved cases rank last
    return candidate_rank(scores, _cfg(rcfg, policy))


def retention_kept(scores: CaseScores, K: int, rcfg: RetentionConfig, policy: str) -> np.ndarray:
    """Kept mask from ``select_active_set``; the exposure key enters as an all-evidenced Q."""
    if policy == "exposure_only":
        scores = replace(scores, Q=ranking(scores, policy, rcfg), evidenced=np.ones(len(scores.case_ids), dtype=bool))
    kept = np.zeros(len(scores.case_ids), dtype=bool)
    kept[select_active_set(scores, K, _cfg(rcfg, policy), None).keep_slots] = True
    return kept


def fill(rank, tie, cohorts, protected, floors, K):
    """select_active_set's fill: protected cases, cohort floors, then the pool, by -rank then ``tie``.

    Returns the kept mask and the floor picks.
    """
    chosen, floor = protected.copy(), np.zeros(len(rank), dtype=bool)
    for c, f in floors.items():
        members = np.flatnonzero((cohorts == c) & ~chosen)
        need = max(0, f - int(chosen[cohorts == c].sum()))
        pick = members[np.lexsort((tie[members], -rank[members]))][:need]
        chosen[pick] = floor[pick] = True
    pool = np.flatnonzero(~chosen)
    chosen[pool[np.lexsort((tie[pool], -rank[pool]))][:K - int(chosen.sum())]] = True
    return chosen, floor


def cut_tied(rank, kept, floor, protected, cohorts) -> np.ndarray:
    """Archived cases another tie order could keep: an equal-ranked, unprotected kept case took the
    slot they competed for (a pool pick, or a floor pick of their own cohort)."""
    rivals = kept & ~protected
    out = np.zeros(len(rank), dtype=bool)
    for j in np.flatnonzero(~kept):
        out[j] = bool((rivals & (rank == rank[j]) & (~floor | (cohorts == cohorts[j]))).any())
    return out


def jaccard(a, b, within=None):
    if within is not None:
        a, b = a & within, b & within
    union = int((a | b).sum())
    return None if union == 0 else int((a & b).sum()) / union


def share(mask, of=None):
    """Mean of ``mask`` (over ``of``); None when empty."""
    mask = np.asarray(mask, dtype=bool) if of is None else np.asarray(mask, dtype=bool)[of]
    return None if mask.size == 0 else float(mask.mean())


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(x, y).statistic)
    return None if np.isnan(rho) else rho


def top_share(A):
    """Share of the total activation mass held by the top 10% of cases (ceil(n / 10))."""
    total = float(A.sum())
    if total <= 0:
        return None
    return float(np.sort(A)[::-1][:-(-len(A) // 10)].sum()) / total


def diagnose(rows: list[dict], manifest: dict, K_spec=0.5, draws: int = 200, tie_seed: int = 0) -> dict:
    """All per-run diagnostics; raises ValueError when the run cannot be selected offline."""
    scfg, rcfg = run_config(manifest)
    scores, n_archived = load_scores(rows, scfg)
    n = len(scores.case_ids)
    if n == 0:
        raise ValueError("no active cases")
    K = min(resolve_K(K_spec, n), n)
    R, A, H, ev = scores.R, scores.A, scores.H, np.asarray(scores.evidenced, dtype=bool)
    cohorts = np.asarray(scores.cohorts)
    labels, codes = np.unique(cohorts, return_inverse=True)
    size = np.bincount(codes, minlength=len(labels))
    protected, floors, _ = protection_requirements(scores, _cfg(rcfg, "provenance_only"), None)
    rng = np.random.default_rng([int(tie_seed), int(rcfg.seed)])
    perms = [rng.permutation(n) for _ in range(draws)]
    total = float(A.sum())
    retrieved = R > 0
    m = {"n": n, "K": K, "n_protected": int(protected.sum()),
         "never_retrieved_share": share(R == 0), "evidenced_share": share(ev),
         "top10_activation_share": top_share(A),
         "spearman_q_a_evidenced": spearman(scores.Q[ev], A[ev]), "evidenced_h0_share": share(H == 0, ev),
         "policies": {}, "jaccard": {}}
    kept = {}
    for p in POLICIES:
        rank = ranking(scores, p, rcfg)
        k, floor = fill(rank, scores.case_ids, cohorts, protected, floors, K)
        if not np.array_equal(k, retention_kept(scores, K, rcfg, p)):
            raise RuntimeError(f"offline fill disagrees with select_active_set for {p}")
        kept[p] = k
        frac = np.bincount(codes[k], minlength=len(labels)) / size
        counts, jac = np.empty((draws, len(labels)), dtype=np.int64), np.empty(draws)
        for d, t in enumerate(perms):
            kr = fill(rank, t, cohorts, protected, floors, K)[0]
            counts[d] = np.bincount(codes[kr], minlength=len(labels))
            jac[d] = jaccard(k, kr)
        entry = {"archived_activation_share": None if total <= 0 else float(A[~k].sum()) / total,
                 "cut_tie_share": share(cut_tied(rank, k, floor, protected, cohorts), ~k),
                 "cohort_kept": {str(c): {"case_id": float(frac[i])} for i, c in enumerate(labels)}}
        if draws:  # integer counts keep tie-free draws exact (sd 0, mean = case-ID fraction)
            mean, sd = counts.mean(0) / size, counts.std(0) / size
            lo, hi = counts.min(0) / size, counts.max(0) / size
            for i, c in enumerate(labels):
                entry["cohort_kept"][str(c)].update(random_mean=float(mean[i]), random_sd=float(sd[i]),
                                                    random_min=float(lo[i]), random_max=float(hi[i]))
            entry.update(tie_shift_max_abs=float(np.abs(frac - mean).max()), random_sd_max=float(sd.max()),
                         random_tie_jaccard_mean=float(jac.mean()))
        if p == "provenance_only":
            m["tie_decided_share"] = share(rank == TIE, ~k)
        m["policies"][p] = entry
    for a, b in PAIRS:
        m["jaccard"][f"{a}|{b}"] = {"all": jaccard(kept[a], kept[b]), "retrieved": jaccard(kept[a], kept[b], retrieved)}
    context = {"n_archived": n_archived, "K_spec": K_spec, "draws": draws, "tie_seed": tie_seed,
               "statistics": {k: getattr(scfg, k) for k in ("smoothing", "min_retrieval_count", "min_activation_mass",
                                                             "bias_normalization", "bias_source")},
               "retention": {k: getattr(rcfg, k, None) for k in ("min_per_cohort", "protect_rare_cohort_below",
                                                                  "boundary_protect_fraction", "seed", "tie_break")}}
    return {"context": context, "metrics": m}


def replay(events: list[dict], rcfg: RetentionConfig) -> dict:
    """Replay logged fill decisions through select_active_set from the logged scores (slot order),
    with the run's own settings. Per logged policy: steps, steps replayed, steps whose every
    decision agrees, disagreeing decisions. A run from older code can disagree by design."""
    groups: dict = {}
    for e in events:
        groups.setdefault((e["run_id"], e["step"]), []).append(e)
    out: dict = {}
    for rs in groups.values():
        policy = str(rs[0].get("policy_version", "")).split("/")[0]
        o = out.setdefault(policy, {"steps": 0, "replayed": 0, "agree": 0, "case_mismatches": 0})
        o["steps"] += 1
        if policy not in REPLAYABLE or rcfg.boundary_protect_fraction > 0:
            continue
        col = lambda k, dt=float: np.array([r[k] for r in rs], dtype=dt)  # noqa: E731
        scores = CaseScores(col("case_id", np.int64), [str(r["cohort"]) for r in rs], col("R"), col("A"), col("C"),
                            col("H"), col("Q"), col("B"), col("evidenced", bool), col("bias"))
        scores.extra["human_protected"] = np.array([bool(r.get("human_protected", False)) for r in rs])
        kept = np.zeros(len(rs), dtype=bool)
        kept[select_active_set(scores, int(rs[0]["capacity_after"]), replace(rcfg, policy=policy), None).keep_slots] = True
        bad = int((kept != np.array([r["action"] == "keep" for r in rs])).sum())
        o["replayed"] += 1
        o["agree"] += int(bad == 0)
        o["case_mismatches"] += bad
    return out


def maintenance_steps(history: dict) -> int:
    """Epochs whose history record carries a maintenance summary."""
    return sum(1 for recs in history.values() if isinstance(recs, list)
               for r in recs if isinstance(r, dict) and r.get("maintenance") is not None)


def find_runs(root: Path, conditions=None, seeds=None):
    for p in sorted(root.rglob("case_statistics.jsonl")):
        d = p.parent
        if not (d / "run_manifest.json").exists():
            continue
        if conditions and not any(fnmatch.fnmatchcase(d.parent.name, c) for c in conditions):
            continue
        if seeds is not None and d.name.startswith("s") and d.name[1:].isdigit() and int(d.name[1:]) not in seeds:
            continue
        yield d


def analyse_run(d: Path, root: Path, K_spec, draws, tie_seed) -> dict:
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(d / "case_statistics.jsonl")
    out = {"task": manifest.get("task", {}).get("name", d.parent.parent.name), "condition": d.parent.name,
           "seed": int(manifest["run"]["seed"]), "run_dir": d.relative_to(root).as_posix()}
    if not rows:
        return {**out, "skipped": "empty case_statistics.jsonl"}
    try:
        res = diagnose(rows, manifest, K_spec, draws, tie_seed)
    except (KeyError, TypeError, ValueError) as e:
        return {**out, "skipped": f"{type(e).__name__}: {e}"}
    ctx = res["context"]
    stats = manifest["resolved_config"]["config"]["statistics"]
    ctx.update(source=stats.get("source"), accumulation=stats.get("accumulation"))
    if (d / "history.json").exists():
        ctx["maintenance_steps"] = maintenance_steps(json.loads((d / "history.json").read_text(encoding="utf-8")))
    events = read_jsonl(d / "case_maintenance.jsonl") if (d / "case_maintenance.jsonl").exists() else []
    if events:
        ctx["replay"] = replay(events, run_config(manifest)[1])
    return {**out, **res}


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def summarize(runs: list[dict]) -> dict:
    """Per task and condition, the mean of every metric over the seeds that define it."""
    groups: dict = {}
    for r in runs:
        if "metrics" in r:
            groups.setdefault(r["task"], {}).setdefault(r["condition"], []).append(r)
    out = {}
    for task, conds in sorted(groups.items()):
        out[task] = {}
        for cond, rs in sorted(conds.items()):
            vals = [flat(r["metrics"]) for r in rs]
            out[task][cond] = {"seeds": sorted(r["seed"] for r in rs),
                               **{k: float(np.mean([v[k] for v in vals if k in v])) for k in sorted(set().union(*vals))}}
    return out


COLUMNS = (("n", "n", "{:.0f}"), ("R=0", "never_retrieved_share", None), ("ev", "evidenced_share", None),
           ("top10", "top10_activation_share", None), ("rho(Q,A)", "spearman_q_a_evidenced", None),
           ("H=0", "evidenced_h0_share", None), ("tie@.5", "tie_decided_share", None),
           ("cut Q", "policies.provenance_only.cut_tie_share", None),
           ("arch Q", "policies.provenance_only.archived_activation_share", None),
           ("arch B", "policies.bias_current.archived_activation_share", None),
           ("arch E", "policies.exposure_only.archived_activation_share", None),
           ("J QB", "jaccard.provenance_only|bias_current", "pair"), ("J QE", "jaccard.provenance_only|exposure_only", "pair"),
           ("J BE", "jaccard.bias_current|exposure_only", "pair"),
           ("dtie Q", "policies.provenance_only.tie_shift_max_abs", None), ("sd Q", "policies.provenance_only.random_sd_max", None))


def _cell(row: dict, key: str, fmt) -> str:
    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    if fmt == "pair":
        return f"{f(row.get(key + '.all'))}/{f(row.get(key + '.retrieved'))}"
    v = row.get(key)
    return "-" if v is None else (fmt or "{:.2f}").format(v)


def markdown(summary: dict, K_spec, seeds_label: str, draws: int, root: str) -> str:
    lines = [f"# CBM diagnostics at K = {K_spec}", "",
             f"Runs `{root}`, seeds {seeds_label}: mean over seeds per condition (k = seeds). Final train-LOO audit, "
             "active cases. Q = provenance_only, B = bias_current, E = exposure-only control (rank by A, never-retrieved "
             "first out), each selected by retention.select_active_set with the run's floors and protections and the "
             "case-ID tie-break.", "",
             "R=0 never-retrieved share; ev evidenced share; top10 share of activation mass A in the top 10% of cases; "
             "rho(Q,A) Spearman among evidenced cases; H=0 share of evidenced cases without harmful support; tie@.5 share "
             "of Q-archived cases at the 0.5 tie score; cut Q share of Q-archived cases an equal-ranked kept case "
             "displaced (decided by the tie-break); arch share of A archived; J Jaccard of kept sets over all cases / "
             f"over R>0 cases; dtie Q largest per-cohort |kept fraction (case-ID tie) - mean over {draws} random "
             "tie-breaks|; sd Q largest per-cohort sd over the random tie-breaks.", ""]
    for task, conds in summary.items():
        lines += [f"## {task}", "", "| condition | k | " + " | ".join(c[0] for c in COLUMNS) + " |",
                  "|---|---:|" + "---:|" * len(COLUMNS)]
        for cond, row in conds.items():
            lines.append(f"| {cond} | {len(row['seeds'])} | " + " | ".join(_cell(row, k, f) for _, k, f in COLUMNS) + " |")
        lines.append("")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", help="t1_run experiment directory (<task>/<condition>/s<seed>/ below it)")
    ap.add_argument("--K", type=parse_K, default=0.5, help="capacity: a float <= 1 is a fraction of the active cases")
    ap.add_argument("--condition", nargs="+", default=None, metavar="GLOB", help="condition directory names to include")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEV_SEEDS), help="default: development seeds 20-25")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--draws", type=int, default=200, help="random tie-break permutations per run")
    ap.add_argument("--tie-seed", type=int, default=0, help="seeds the permutations together with the run seed")
    ap.add_argument("--json", help="write runs and summary as JSON")
    ap.add_argument("--md", help="write the per-task tables as markdown (default: print)")
    args = ap.parse_args(argv)
    root = Path(args.runs)
    seeds = None if args.all_seeds else set(args.seeds)
    runs = [analyse_run(d, root, args.K, args.draws, args.tie_seed) for d in find_runs(root, args.condition, seeds)]
    runs = [r for r in runs if seeds is None or r["seed"] in seeds]
    summary = summarize(runs)
    seeds_label = "all" if seeds is None else ", ".join(map(str, sorted(seeds)))
    md = markdown(summary, args.K, seeds_label, args.draws, str(root))
    report = {"runs_dir": str(root), "K": args.K, "seeds": None if seeds is None else sorted(seeds), "draws": args.draws,
              "tie_seed": args.tie_seed, "conditions": args.condition, "summary": summary,
              "skipped": [{k: r[k] for k in ("run_dir", "skipped")} for r in runs if "skipped" in r],
              "runs": [r for r in runs if "metrics" in r]}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1, default=_json_default), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    else:
        print(md)
    print(f"{len(report['runs'])} runs, {len(report['skipped'])} skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

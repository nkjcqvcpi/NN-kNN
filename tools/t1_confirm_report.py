"""Pre-registered T1 confirmation analysis (docs/T1_PREREG_20261013.md).

    python tools/t1_confirm_report.py --prereg configs/t1/prereg_20261013.json --prereg-sha256 <hex>
        --runs <neural-cbr experiment dirs...> --baselines <grid or evaluate output dirs...>
        --final <arm::name of F> --out <dir> [--groups legacy transfer]

Error is 1 - accuracy (adapted output; pre-adaptation for retrieval-only rows) or
train-standardized RMSE on the test stream. Per task the effect is the mean over
seed-paired runs of log((e_F + eps) / (e_ref + eps)), eps = 0.5 / n_test
(classification) or 0 (regression), with within-task variance var / n_seeds. Tasks
are pooled by random-effects meta-analysis (REML tau^2, Hartung-Knapp interval with
t(k-1)); GMR = exp(pooled effect), < 1 favours F. E1/E2 are one-sided superiority
tests (Holm-significant over {E1, E2, E3a, E3b}, GMR <= 0.97, E2's guards); E3a/E3b
are non-inferiority at the pre-registered margin (H0: GMR >= 1 + delta), decided by
their Holm-adjusted one-sided p (prereg addendum 1); the unadjusted one-sided 95%
upper bound is reported alongside. Secondary analyses
(ECE_T, flips, six-method ranks) use Benjamini-Hochberg. Writes ``report.json``
(every number) and ``report.md``.

Refused: a prereg file whose sha256 differs from --prereg-sha256; an analysed row or
baseline entry on a seed outside the evaluation seeds (plus the small-task seeds for
small tasks); a bare small-task name shared by several analysed tasks; a declared
small task whose test set has 100 or more rows; duplicate rows that differ; rows from
different run source fingerprints (unless --allow-mixed-code); a neural-CBR
comparison (E1 or E2 on one task) whose arms ran on different device types (unless
--allow-mixed-device); missing primary evidence, including an undeclared non-OpenML
task with fewer than 100 test rows (unless --allow-incomplete).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ALPHA = 0.05  # Holm family-wise level over {E1, E2, E3a, E3b}
BH_Q = 0.10  # Benjamini-Hochberg level for secondary analyses
SENSITIVITY_DELTAS = (0.05, 0.20)  # descriptive only
ECE_MARGIN = 0.02  # secondary: median ECE_T <= pre-adaptation ECE + 0.02
FLIP_FACTOR = 2.0  # secondary: wrong-to-correct >= 2 x correct-to-wrong
GROUP_KEYS = {"legacy": "primary_legacy", "transfer": "primary_transfer"}
RANK_FAMILIES = ("catboost", "tabm", "mlp", "knn")
LABELS = {"catboost": "CatBoost", "tabm": "TabM", "mlp": "MLP", "knn": "kNN"}
EXTRA_METRIC_TASKS = ("classification__tabarena_363623", "classification__tabarena_363711")  # churn, MIC
EXTRA_METRIC_PREFIXES = ("test_log_loss", "test_logloss", "test_macro_f1")
PRE = "@pre"  # suffix of a condition's pre-adaptation error series
NAN = float("nan")
SMALL_TEST_ROWS = 100  # prereg: small tasks are non-OpenML tasks with fewer test rows than this
UNKNOWN_DEVICE = "unknown"
DECISION_RULES = {
    "superiority": "Holm-adjusted one-sided p <= alpha and GMR <= gmr_max (E2: and every evaluable guard "
                   "holds); with both groups, no claim if the transfer group's 95% CI lies above 1",
    "noninferiority": "Holm-adjusted one-sided non-inferiority p (H0: GMR >= 1 + delta) <= alpha "
                      "(prereg addendum 1); the unadjusted one-sided 95% upper bound is reported",
}


# ----------------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Task:
    key: str  # <task_type>__<name>, the grid's job directory name
    name: str
    task_type: str
    group: str
    small: bool
    seeds: tuple[int, ...]  # allowed and expected evaluation seeds


@dataclass(frozen=True)
class Conditions:
    final: str
    retrieval_only: str
    default: str


@dataclass
class Evidence:
    err: dict[tuple[str, str], dict[int, float]] = field(default_factory=dict)  # (task, method) -> seed -> error
    rows: dict[tuple[str, str], dict[int, dict]] = field(default_factory=dict)  # (task, condition) -> seed -> row
    n_test: dict[tuple[str, int], int] = field(default_factory=dict)
    n_test_baseline: dict[tuple[str, int], int] = field(default_factory=dict)
    devices: dict[tuple[str, str], dict[int, str]] = field(default_factory=dict)  # (task, condition) -> seed -> device type
    provenance: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def put(self, task: str, method: str, seed: int, value: float) -> None:
        self.err.setdefault((task, method), {})[seed] = value

    def eps(self, task: Task, seed: int) -> float:
        return 0.5 / self.n_test[task.key, seed] if task.task_type == "classification" else 0.0


def condition_id(row: dict) -> str:
    """'<arm>::<name>' as in tools/t1_schedule_search.condition_id (kept torch-free here)."""
    arm = row.get("arm", "")
    label = row["condition"]
    name = label[len(arm) + 2:] if arm and label.startswith(arm + "__") else label
    if name.endswith("retrieval_only"):
        name = "retrieval_only"
    elif "schedule" not in row:  # frozen-core reuse
        name = f"frozen_reuse__{row['loss']}"
    return f"{arm}::{name}" if arm else name


def row_task(row: dict) -> str:
    name = row.get("dataset_name") or row["dataset"].removeprefix(row["task_type"] + "__")
    return f"{row['task_type']}__{name}"


def row_error(row: dict, output: str = "post") -> float:
    if condition_id(row).endswith("retrieval_only"):
        output = "pre"
    if row["task_type"] == "classification":
        return 1.0 - float(row[f"test_accuracy_{output}"])
    return float(row[f"test_rmse_{output}"])


def baseline_error(entry: dict, task_type: str) -> float:
    m = entry["test_metrics"]
    return 1.0 - float(m["accuracy"]) if task_type == "classification" else float(m["rmse_standardized"])


def check_error(value: float, task: Task, where: str) -> float:
    if not math.isfinite(value) or value < 0 or (task.task_type == "regression" and value == 0):
        raise SystemExit(f"invalid error {value} for {task.key} in {where}")
    return value


def load_prereg(path: Path, sha256: str) -> tuple[dict, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256.strip().lower():
        raise SystemExit(f"prereg sha256 mismatch: {path} is {digest}, --prereg-sha256 is {sha256}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def small_task_names(items: list[str], prereg: dict) -> set[str]:
    """CLI names/keys; an item ending in .json is a list or {"small_tasks": [...]} (e.g. a seed audit)."""
    names = set(prereg.get("small_tasks", []))
    for item in items:
        if item.endswith(".json"):
            data = json.loads(Path(item).read_text(encoding="utf-8"))
            data = data.get("small_tasks", []) if isinstance(data, dict) else data
            names |= {f"{d['task_type']}__{d['name']}" if isinstance(d, dict) else d for d in data}
        else:
            names.add(item)
    return names


def is_openml(name: str) -> bool:
    return name.startswith(("tabarena_", "openml_"))


def plan_tasks(prereg: dict, groups: list[str], small: set[str]) -> tuple[list[Task], list[str]]:
    seeds = prereg["seeds"]
    base, extra = set(seeds["evaluation"]), set(seeds.get("evaluation_small_tasks", []))
    tasks, matched, listed_all = [], set(), []
    if len(set(groups)) != len(groups):
        raise SystemExit(f"repeated group in {groups}")
    for g in groups:
        listed = prereg["tasks"].get(GROUP_KEYS.get(g, g))
        if not isinstance(listed, list) or not all(isinstance(t, dict) for t in listed):
            raise SystemExit(f"group {g} has no task list in the prereg; task groups: {sorted(prereg['tasks'])}")
        listed_all += [(g, t) for t in listed]
    # A bare name such as 'wine' can name a classification and a regression task.
    keys_by_name: dict[str, list[str]] = {}
    for _, t in listed_all:
        keys_by_name.setdefault(t["name"], []).append(f"{t['task_type']}__{t['name']}")
    if ambiguous := {n: keys_by_name[n] for n in sorted(small) if len(keys_by_name.get(n, [])) > 1}:
        raise SystemExit(f"small-task name(s) {sorted(ambiguous)} match several analysed tasks {ambiguous}; "
                         "name small tasks by their <task_type>__<name> key")
    for g, t in listed_all:
        key = f"{t['task_type']}__{t['name']}"
        hit = {key, t["name"]} & small
        if hit and is_openml(t["name"]):
            raise SystemExit(f"{key} is an OpenML task; the prereg's extra seeds are for non-OpenML tasks only")
        matched |= hit
        tasks.append(Task(key, t["name"], t["task_type"], g, bool(hit), tuple(sorted(base | (extra if hit else set())))))
    if len({t.key for t in tasks}) != len(tasks):
        raise SystemExit("a task appears in more than one selected group")
    return tasks, sorted(small - matched)


def device_type(value) -> str:
    """'xpu:0' -> 'xpu'; a manifest without a device (older runs) -> 'unknown'."""
    text = "" if value is None else str(value).strip().lower()
    return text.split(":", 1)[0] if text and text != "none" else UNKNOWN_DEVICE


def load_runs(ev: Evidence, dirs: list[Path], tasks: list[Task], cond: Conditions, allow_mixed_code: bool,
              allow_mixed_device: bool = False) -> None:
    by_key = {t.key: t for t in tasks}
    wanted = {cond.final, cond.retrieval_only, cond.default}
    counts = {"rows_analysed": 0, "rows_ignored": 0, "duplicate_rows": 0, "run_manifests_found": 0,
              "run_manifests_absent": 0}
    fingerprints: dict[str, dict[str, int]] = {}
    bad_seed, conflicts = [], []
    for d in dirs:
        shards = sorted(d.glob("runs__*.jsonl"))
        if not shards:
            raise SystemExit(f"no runs__*.jsonl rows in {d}")
        for shard in shards:
            for line in shard.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                t, c, s = by_key.get(row_task(row)), condition_id(row), int(row["seed"])
                if t is None or c not in wanted:
                    counts["rows_ignored"] += 1
                    continue
                if s not in t.seeds:
                    bad_seed.append(f"{t.key} {c} seed {s} ({shard})")
                    continue
                seen = ev.rows.setdefault((t.key, c), {})
                if s in seen:  # only bitwise-identical reruns are allowed
                    counts["duplicate_rows"] += 1
                    same = {k: v for k, v in seen[s].items() if k.startswith(("test_", "val_"))}
                    if same != {k: v for k, v in row.items() if k.startswith(("test_", "val_"))}:
                        conflicts.append(f"{t.key} {c} seed {s} ({shard})")
                    continue
                seen[s] = row
                counts["rows_analysed"] += 1
                if row.get("sync_no_eligible_checkpoint"):  # untrained adapter: the guard rejected every epoch
                    ev.warnings.append(f"{t.key} {c} seed {s}: retrieval guard left no eligible checkpoint; "
                                       "post-adaptation metrics come from the epoch-0 state")
                ev.put(t.key, c, s, check_error(row_error(row), t, str(shard)))
                if c != cond.retrieval_only:
                    ev.put(t.key, c + PRE, s, check_error(row_error(row, "pre"), t, str(shard)))
                if row.get("test_n") is not None:
                    if ev.n_test.setdefault((t.key, s), int(row["test_n"])) != int(row["test_n"]):
                        raise SystemExit(f"{t.key} seed {s}: rows disagree on test_n; the splits differ")
                manifest = d / row["dataset"] / row["condition"] / f"s{s}" / "run_manifest.json"
                device = UNKNOWN_DEVICE
                if manifest.exists():
                    run = json.loads(manifest.read_text(encoding="utf-8"))["run"]
                    fp = fingerprints.setdefault(str(run.get("source_fingerprint_sha256")), {})
                    fp[c] = fp.get(c, 0) + 1
                    device = device_type(run.get("device"))
                    counts["run_manifests_found"] += 1
                else:
                    counts["run_manifests_absent"] += 1
                ev.devices.setdefault((t.key, c), {})[s] = device
    if bad_seed:
        raise SystemExit(f"{len(bad_seed)} row(s) on seeds outside the pre-registered evaluation seeds, e.g. {bad_seed[:5]}")
    if conflicts:
        raise SystemExit(f"{len(conflicts)} duplicate row(s) differ from an earlier run, e.g. {conflicts[:5]}")
    if len(fingerprints) > 1 and not allow_mixed_code:
        raise SystemExit(f"rows come from {len(fingerprints)} run source fingerprints {sorted(fingerprints)}; "
                         "pass --allow-mixed-code to analyse them together")
    if not any(c == cond.final for _, c in ev.rows):
        raise SystemExit(f"no rows for the final condition {cond.final}")
    ev.provenance.update(counts, source_fingerprints=fingerprints)
    check_devices(ev, tasks, cond, allow_mixed_device)


def check_devices(ev: Evidence, tasks: list[Task], cond: Conditions, allow_mixed_device: bool) -> None:
    """The prereg forbids pooling CPU and XPU arms within a comparison: on each task, the
    neural-CBR arms of E1 (F, retrieval-only) and of E2 and its guards (F, default) must
    share one device type over all seeds. Rows whose manifest records no device are unknown."""
    pairs = {"E1": (cond.final, cond.retrieval_only), "E2": (cond.final, cond.default)}
    totals: dict[str, int] = {}
    for series in ev.devices.values():
        for dev in series.values():
            totals[dev] = totals.get(dev, 0) + 1
    by_task, mixed = {}, []
    for t in tasks:
        seen = {c: sorted(set(ev.devices.get((t.key, c), {}).values())) for c in dict.fromkeys(sum(pairs.values(), ()))}
        if any(seen.values()):
            by_task[t.key] = {c: v for c, v in seen.items() if v}
        for name, pair in pairs.items():
            known = sorted({d for c in pair for d in seen[c] if d != UNKNOWN_DEVICE})
            if len(known) > 1:
                mixed.append(f"{t.key} {name}: {known}")
    unknown = totals.get(UNKNOWN_DEVICE, 0)
    ev.provenance.update(devices=totals, devices_by_task=by_task,
                         device_check={"mixed_comparisons": mixed, "allow_mixed_device": allow_mixed_device,
                                       "rows_without_device": unknown})
    if mixed and not allow_mixed_device:
        raise SystemExit(f"{len(mixed)} comparison(s) pool neural-CBR arms run on different device types, which the "
                         f"prereg forbids, e.g. {mixed[:5]}; pass --allow-mixed-device to analyse them anyway")
    if mixed:
        ev.warnings.append(f"analysed with --allow-mixed-device: {len(mixed)} comparison(s) pool device types, e.g. {mixed[:3]}")
    if unknown:
        ev.warnings.append(f"no recorded device for {unknown} of {sum(totals.values())} neural-CBR rows; "
                           "the per-comparison CPU/XPU check covers only rows whose run manifest records one")


def load_baselines(ev: Evidence, dirs: list[Path], tasks: list[Task], families: list[str]) -> None:
    files, skipped, bad_seed, sources = [], 0, [], {}
    for d in dirs:
        found = 0
        for t in tasks:
            for fam in families:
                for name in ("completed.json", "evaluation.json"):
                    path = d / t.key / fam / name
                    if not path.exists():
                        continue
                    found += 1
                    files.append(str(path))
                    data = json.loads(path.read_text(encoding="utf-8"))
                    for e in data if isinstance(data, list) else data.get("evaluation", []):
                        if e.get("status", "completed") != "completed" or "test_metrics" not in e:
                            skipped += 1
                            continue
                        s = int(e["seed"])
                        if s not in t.seeds:
                            bad_seed.append(f"{t.key} {fam} seed {s} ({path})")
                            continue
                        value = check_error(baseline_error(e, t.task_type), t, str(path))
                        old = ev.err.get((t.key, f"baseline:{fam}"), {}).get(s)
                        if old is not None and old != value:
                            raise SystemExit(f"{t.key} {fam} seed {s}: baseline outputs disagree ({path})")
                        ev.put(t.key, f"baseline:{fam}", s, value)
                        n = (e.get("dataset") or {}).get("n_test")
                        if n is not None:
                            ev.n_test_baseline[t.key, s] = int(n)
                        per_family, sha = sources.setdefault(fam, {}), str(e.get("source_sha256"))
                        per_family[sha] = per_family.get(sha, 0) + 1
        if not found:
            raise SystemExit(f"no <task>/<family>/completed.json or evaluation.json for the analysed tasks under {d}")
    if bad_seed:
        raise SystemExit(f"{len(bad_seed)} baseline entr(ies) on seeds outside the evaluation seeds, e.g. {bad_seed[:5]}")
    ev.provenance.update(baseline_files=files, baseline_entries_not_completed=skipped, baseline_source_sha256=sources)


def resolve_n_test(ev: Evidence, tasks: list[Task], fallback: dict) -> None:
    """n_test per (task, seed): rows' test_n, else the baselines' split description, else the fallback map."""
    counts, missing = {"rows": 0, "baselines": 0, "fallback": 0}, []
    for t in tasks:
        if t.task_type != "classification":
            continue
        for s in sorted({s for (k, _), series in ev.err.items() if k == t.key for s in series}):
            b = ev.n_test_baseline.get((t.key, s))
            if (t.key, s) in ev.n_test:
                counts["rows"] += 1
                if b is not None and b != ev.n_test[t.key, s]:
                    ev.warnings.append(f"{t.key} seed {s}: baseline n_test {b} != neural-CBR test_n {ev.n_test[t.key, s]}")
                continue
            if b is not None:
                ev.n_test[t.key, s] = b
                counts["baselines"] += 1
                continue
            fb = fallback.get(t.key, fallback.get(t.name))
            fb = fb.get(str(s)) if isinstance(fb, dict) else fb
            if fb is None:
                missing.append(f"{t.key} seed {s}")
                continue
            ev.n_test[t.key, s] = int(fb)
            counts["fallback"] += 1
    if missing:
        raise SystemExit(f"no n_test for {len(missing)} classification run(s), e.g. {missing[:5]}; pass --n-test")
    ev.provenance["n_test_sources"] = counts


def check_small_tasks(ev: Evidence, tasks: list[Task], allow_incomplete: bool) -> None:
    """Cross-check the small-task declaration against the test-set sizes that were read.

    The prereg's small tasks are the non-OpenML tasks with fewer than 100 test rows. A
    declared small task with a larger test set is refused; an undeclared one would lack
    its pre-registered small-task seeds, so it is incomplete evidence."""
    known: dict[str, list[int]] = {}
    for (key, _), n in [*ev.n_test_baseline.items(), *ev.n_test.items()]:
        known.setdefault(key, []).append(n)
    wrong, undeclared = [], []
    for t in tasks:
        if not (sizes := known.get(t.key)):
            continue
        if t.small and max(sizes) >= SMALL_TEST_ROWS:
            wrong.append(f"{t.key} ({max(sizes)} test rows)")
        elif not t.small and not is_openml(t.name) and max(sizes) < SMALL_TEST_ROWS:
            undeclared.append(f"{t.key} ({max(sizes)} test rows)")
    ev.provenance["small_task_check"] = {"declared_but_large": wrong, "small_but_undeclared": undeclared,
                                         "tasks_without_test_size": [t.key for t in tasks if t.key not in known]}
    if wrong:
        raise SystemExit(f"declared small task(s) with {SMALL_TEST_ROWS} or more test rows: {wrong}; the prereg's "
                         "extra seeds are for tasks with fewer")
    if undeclared:
        msg = (f"task(s) with fewer than {SMALL_TEST_ROWS} test rows not declared small, so their pre-registered "
               f"small-task seeds are not analysed: {undeclared}")
        if not allow_incomplete:
            raise SystemExit(msg + "; pass --small-tasks (e.g. the seed audit JSON) or --allow-incomplete")
        ev.warnings.append(msg)


def missing_evidence(ev: Evidence, tasks: list[Task], methods: list[str]) -> list[dict]:
    out = []
    for t in tasks:
        for m in methods:
            have = ev.err.get((t.key, m), {})
            if gap := [s for s in t.seeds if s not in have]:
                out.append({"task": t.key, "method": m, "missing_seeds": gap})
    return out


# ----------------------------------------------------------------------------- statistics

@dataclass(frozen=True)
class Pooled:
    """Random-effects pooled log ratio: REML tau^2, Hartung-Knapp standard error, t(k-1)."""
    k: int
    mu: float
    se: float
    tau2: float

    def p_below(self, c: float = 0.0) -> float:
        """One-sided p-value of H0: mu >= c against H1: mu < c."""
        if self.k < 2:
            return NAN
        if self.se == 0:
            return 0.0 if self.mu < c else 1.0
        return float(stats.t.cdf((self.mu - c) / self.se, self.k - 1))

    def upper(self, level: float = 0.95) -> float:
        return self.mu + float(stats.t.ppf(level, self.k - 1)) * self.se if self.k >= 2 else NAN

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        h = float(stats.t.ppf(0.5 + level / 2, self.k - 1)) * self.se if self.k >= 2 else NAN
        return self.mu - h, self.mu + h

    def summary(self) -> dict:
        lo, hi = self.ci()
        return {"k": self.k, "log_gmr": self.mu, "se_hk": self.se, "tau2": self.tau2, "df": self.k - 1,
                "log_ci95": [lo, hi], "gmr": math.exp(self.mu) if math.isfinite(self.mu) else NAN,
                "gmr_ci95": [_exp(lo), _exp(hi)], "gmr_upper95_one_sided": _exp(self.upper())}


def _exp(x: float) -> float:
    return math.exp(x) if math.isfinite(x) else NAN


def reml_tau2(y, v) -> float:
    """REML between-task variance: maximize the restricted likelihood over tau^2 >= 0."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    if len(y) < 2:
        return 0.0

    def nll(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * float(np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))

    hi = 10.0 * float(np.var(y, ddof=1)) + float(v.max())
    grid = np.concatenate([[0.0], np.geomspace(hi * 1e-10, hi, 101)])
    vals = [nll(g) for g in grid]
    i = int(np.argmin(vals))
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded",
                                   options={"xatol": max(hi_b * 1e-12, 1e-300), "maxiter": 1000})
    # The bounded search never lands exactly on 0; the grid's exact 0 wins at the boundary.
    return float(res.x) if res.fun < vals[i] else float(grid[i])


def pool(y, v) -> Pooled:
    y, v = np.asarray(y, float), np.asarray(v, float)
    if len(y) < 2:
        return Pooled(len(y), float(y[0]) if len(y) else NAN, NAN, NAN)
    if (v <= 0).any():
        raise ValueError("within-task variances must be positive")
    tau2 = reml_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu) ** 2)) / (len(y) - 1)
    return Pooled(len(y), mu, math.sqrt(q / float(np.sum(w))), tau2)


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=lambda k: (p[k], k))
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def bh(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=lambda k: (p[k], k))
    out, run = {}, 1.0
    for i in range(len(keys) - 1, -1, -1):
        run = min(run, len(keys) * p[keys[i]] / (i + 1))
        out[keys[i]] = min(1.0, run)
    return out


def _wilcoxon(d, alternative: str) -> float:
    d = np.asarray(d, float)
    if len(d) < 2 or not np.any(d != 0):
        return NAN
    return float(stats.wilcoxon(d, alternative=alternative).pvalue)


# ----------------------------------------------------------------------------- analysis

def compare(ev: Evidence, tasks: list[Task], a: str, b: str, min_var: float | None = None) -> dict:
    """Seed-paired log error ratios of ``a`` over ``b`` per task, pooled over tasks.

    A task whose paired ratios do not vary has zero within-task variance; it is floored
    at ``min_var`` or, by default, at the smallest positive within-task variance here.
    """
    per = []
    for t in tasks:
        fa, fb = ev.err.get((t.key, a), {}), ev.err.get((t.key, b), {})
        if not (seeds := sorted(set(fa) & set(fb))):
            continue
        d = np.array([math.log((fa[s] + ev.eps(t, s)) / (fb[s] + ev.eps(t, s))) for s in seeds])
        per.append({"task": t.key, "group": t.group, "n_seeds": len(d), "log_ratio": float(d.mean()),
                    "within_var": float(d.var(ddof=1)) / len(d) if len(d) > 1 else NAN, "ratio": math.exp(d.mean()),
                    "error": float(np.mean([fa[s] for s in seeds])), "reference_error": float(np.mean([fb[s] for s in seeds]))})
    usable = [p for p in per if p["n_seeds"] > 1]
    raw = np.array([p["within_var"] for p in usable])
    floor = min_var if min_var is not None else (float(raw[raw > 0].min()) if (raw > 0).any() else 1e-12)
    for p in usable:
        p["variance_floored"] = bool(p["within_var"] < floor)
    pooled = pool([p["log_ratio"] for p in usable], np.maximum(raw, floor))
    return {"method": a, "reference": b, "pooled": pooled, "variance_floor": floor, "per_task": per,
            "tasks_without_pairs": [t.key for t in tasks if t.key not in {p["task"] for p in per}],
            "tasks_single_seed": [p["task"] for p in per if p["n_seeds"] < 2]}


def endpoint_refs(prereg: dict, cond: Conditions) -> dict[str, str]:
    spec = prereg["endpoints"]
    return {"E1": cond.retrieval_only, "E2": cond.default,
            "E3a": f"baseline:{spec['E3a']['vs']}", "E3b": f"baseline:{spec['E3b']['vs']}"}


def endpoints(ev: Evidence, tasks: list[Task], cond: Conditions, prereg: dict, min_var: float | None) -> dict:
    out = {}
    for name, ref in endpoint_refs(prereg, cond).items():
        spec = prereg["endpoints"][name]
        c = compare(ev, tasks, cond.final, ref, min_var)
        margin = 0.0 if spec["test"] == "superiority" else math.log1p(spec["delta"])
        out[name] = {"test": spec["test"], "reference": ref, **c["pooled"].summary(), "margin_log": margin,
                     "p_one_sided": c["pooled"].p_below(margin), "variance_floor": c["variance_floor"],
                     "tasks_without_pairs": c["tasks_without_pairs"], "tasks_single_seed": c["tasks_single_seed"],
                     "per_task": c["per_task"]}
        if spec["test"] != "superiority":
            out[name]["delta"] = spec["delta"]
            out[name]["sensitivity"] = {f"{d:g}": {"p_one_sided": c["pooled"].p_below(math.log1p(d)),
                                                   "upper_bound_below_margin": out[name]["gmr_upper95_one_sided"] < 1 + d}
                                        for d in sorted({*SENSITIVITY_DELTAS, spec["delta"]})}
    holm_p = holm({k: r["p_one_sided"] if math.isfinite(r["p_one_sided"]) else 1.0 for k, r in out.items()})
    for k, r in out.items():
        r["p_holm"] = holm_p[k]
    return out


def ratio_of_means(ev: Evidence, tasks: list[Task], a: str, b: str, key: str) -> list[dict]:
    out = []
    for t in tasks:
        ra, rb = ev.rows.get((t.key, a), {}), ev.rows.get((t.key, b), {})
        seeds = [s for s in sorted(set(ra) & set(rb)) if ra[s].get(key) is not None and rb[s].get(key) is not None]
        if seeds:
            ma, mb = float(np.mean([ra[s][key] for s in seeds])), float(np.mean([rb[s][key] for s in seeds]))
            out.append({"task": t.key, "n_seeds": len(seeds), "value": ma, "reference_value": mb,
                        "ratio": ma / mb if mb > 0 else (1.0 if ma == mb else math.inf)})
    return out


def guards(ev: Evidence, tasks: list[Task], cond: Conditions, prereg: dict, min_var: float | None) -> dict:
    """E2 guards: F's pre-adaptation GMR vs retrieval-only, locality and correction size vs the default."""
    g = prereg["endpoints"]["E2"]["guards"]
    pre_cmp = compare(ev, tasks, cond.final + PRE, cond.retrieval_only, min_var)
    pre = pre_cmp["pooled"].summary()
    loc = ratio_of_means(ev, tasks, cond.final, cond.default, "test_locality_input")
    cor = ratio_of_means(ev, tasks, cond.final, cond.default, "test_correction_l1_mean")
    with np.errstate(divide="ignore", invalid="ignore"):  # a zero or infinite ratio stays 0 / inf
        loc_value = float(np.exp(np.mean(np.log([x["ratio"] for x in loc])))) if loc else NAN
    cor_value = float(np.median([x["ratio"] for x in cor])) if cor else NAN
    out = {
        "pre_adaptation": {"method": "REML/Hartung-Knapp GMR of F's pre-adaptation error vs retrieval-only",
                           "value": pre["gmr"], "limit": g["pre_adaptation_gmr_max"], "evaluated": pre["k"] >= 2,
                           "holds": pre["gmr"] <= g["pre_adaptation_gmr_max"], "pooled": pre,
                           "per_task": pre_cmp["per_task"]},
        "locality": {"method": "geometric mean over tasks of the ratio of seed-mean test_locality_input, F / default",
                     "value": loc_value, "limit": g["locality_ratio_max"], "evaluated": bool(loc),
                     "holds": loc_value <= g["locality_ratio_max"] if loc else None, "per_task": loc},
        "correction": {"method": "median over tasks of the ratio of seed-mean test_correction_l1_mean, F / default",
                       "value": cor_value, "limit": g["correction_ratio_max"], "evaluated": bool(cor),
                       "holds": cor_value <= g["correction_ratio_max"] if cor else None, "per_task": cor},
    }
    # The locality guard applies only when locality was recorded; the others must be evaluable.
    out["hold"] = bool(out["pre_adaptation"]["holds"] and out["correction"]["holds"]
                       and out["locality"]["holds"] is not False)
    return out


def decide(primary: dict, guard: dict, transfer: dict | None, prereg: dict) -> None:
    """Pre-registered decisions and the prereg's wording, mechanically, in place."""
    for name, r in primary.items():
        spec = prereg["endpoints"][name]
        r["holm_significant"] = r["p_holm"] <= ALPHA
        r["decision_rule"] = DECISION_RULES["superiority" if spec["test"] == "superiority" else "noninferiority"]
        if spec["test"] == "superiority":
            crit = {"holm_significant": r["holm_significant"], "gmr_at_most_max": r["gmr"] <= spec["gmr_max"]}
            if transfer is not None:
                crit["transfer_not_opposite"] = not transfer[name]["gmr_ci95"][0] > 1.0
            if name == "E2":
                crit["guards_hold"] = guard["hold"]
        else:  # prereg addendum 1: E3 is in the Holm family; the unadjusted bound is reported only
            r["upper_bound_below_margin"] = r["gmr_upper95_one_sided"] < 1 + spec["delta"]
            crit = {"holm_significant": r["holm_significant"]}
        r["criteria"], r["supported"] = crit, all(crit.values())
        r["wording"] = wording(name, r, spec, guard)


def wording(name: str, r: dict, spec: dict, guard: dict) -> str:
    g, (lo, hi) = r["gmr"], r["gmr_ci95"]
    if r["k"] < 2 or not math.isfinite(g):
        return "not evaluable (fewer than two tasks with paired evidence)"
    ci = f"GMR {g:.3f} [95% CI {lo:.3f}, {hi:.3f}]"
    label = LABELS.get(spec["vs"], spec["vs"])
    if r["supported"]:
        return {"E1": f"reuse lowers error vs retrieval-only on the same tuned core: {ci}",
                "E2": f"the selected schedule lowers error vs the alternating default: {ci}"}.get(
            name, f"non-inferior to tuned {label} at delta = {spec.get('delta', NAN):g}: {ci}, "
                  f"one-sided 95% upper bound {r['gmr_upper95_one_sided']:.3f}")
    if name == "E1":
        return "no detectable gain from reuse on tuned cores"
    if name == "E2":
        return "gate 5 fails" if not guard["hold"] else "no detectable schedule effect"
    if g < 1:  # F's error is lower but the bound reaches the margin: "trails" would misstate the direction
        return (f"non-inferiority to tuned {label} not shown at delta = {spec['delta']:g}: {ci}, "
                f"one-sided 95% upper bound {r['gmr_upper95_one_sided']:.3f}")
    return f"trails tuned {label} by {100 * (g - 1):+.1f}% [95% CI {100 * (lo - 1):+.1f}%, {100 * (hi - 1):+.1f}%]"


def secondary(ev: Evidence, tasks: list[Task], cond: Conditions) -> dict:
    cls = [t for t in tasks if t.task_type == "classification"]
    ece = []
    for t in cls:
        rs = [r for r in ev.rows.get((t.key, cond.final), {}).values()
              if r.get("test_ece_post_T") is not None and r.get("test_ece_pre") is not None]
        if rs:
            ece.append({"task": t.key, "n_seeds": len(rs), "ece_post_T": float(np.mean([r["test_ece_post_T"] for r in rs])),
                        "ece_pre": float(np.mean([r["test_ece_pre"] for r in rs])),
                        "ece_post": float(np.mean([r.get("test_ece_post", NAN) for r in rs]))})
    out = {"ece": {"available": bool(ece), "per_task": ece}}
    if ece:
        mt, mp = float(np.median([e["ece_post_T"] for e in ece])), float(np.median([e["ece_pre"] for e in ece]))
        out["ece"].update(tasks=len(ece), median_ece_post_T=mt, median_ece_pre=mp,
                          median_ece_post_uncalibrated=float(np.median([e["ece_post"] for e in ece])),
                          holds=mt <= mp + ECE_MARGIN,
                          test="Wilcoxon signed-rank over tasks of ECE_T - ECE_pre - 0.02, H1: < 0",
                          p=_wilcoxon([e["ece_post_T"] - e["ece_pre"] - ECE_MARGIN for e in ece], "less"))
    flips = []
    for t in cls:
        rs = [r for r in ev.rows.get((t.key, cond.final), {}).values() if "test_flip_wrong_to_correct" in r]
        if rs:
            flips.append({"task": t.key, "n_seeds": len(rs),
                          "wrong_to_correct": int(sum(r["test_flip_wrong_to_correct"] for r in rs)),
                          "correct_to_wrong": int(sum(r["test_flip_correct_to_wrong"] for r in rs))})
    out["flips"] = {"available": bool(flips), "per_task": flips}
    if flips:
        w, c = sum(f["wrong_to_correct"] for f in flips), sum(f["correct_to_wrong"] for f in flips)
        active = [f for f in flips if f["wrong_to_correct"] + f["correct_to_wrong"] > 0]
        # No flips at all: the ratio is undefined and the criterion is not evaluable (not vacuously met).
        out["flips"].update(wrong_to_correct=w, correct_to_wrong=c, ratio=w / c if c else (math.inf if w else NAN),
                            holds=w >= FLIP_FACTOR * c if w + c else None,
                            test="Wilcoxon signed-rank over tasks with flips of log((W+0.5)/(C+0.5)) - log 2, H1: > 0",
                            p=_wilcoxon([math.log((f["wrong_to_correct"] + .5) / (f["correct_to_wrong"] + .5))
                                         - math.log(FLIP_FACTOR) for f in active], "greater"))
    methods = [(LABELS[f], f"baseline:{f}") for f in RANK_FAMILIES] + [
        ("NN-kNN retrieval-only", cond.retrieval_only), ("F", cond.final)]
    table, missing = [], []
    for t in tasks:
        series = [ev.err.get((t.key, m), {}) for _, m in methods]
        seeds = sorted(set.intersection(*(set(s) for s in series)))
        if not seeds:
            missing.append(t.key)
            continue
        means = [float(np.mean([s[x] for x in seeds])) for s in series]
        table.append({"task": t.key, "n_seeds": len(seeds), "errors": dict(zip([m for m, _ in methods], means)),
                      "ranks": dict(zip([m for m, _ in methods], stats.rankdata(means).tolist()))})
    out["ranks"] = {"tasks": len(table), "tasks_missing_a_method": missing, "per_task": table,
                    "mean_rank": {m: float(np.mean([r["ranks"][m] for r in table])) if table else NAN for m, _ in methods},
                    "test": "Friedman over tasks", "p": NAN}
    if len(table) >= 2:
        with np.errstate(all="ignore"):
            out["ranks"]["p"] = float(stats.friedmanchisquare(*[[r["errors"][m] for r in table] for m, _ in methods]).pvalue)
    extra = []
    for t in tasks:
        if t.key in EXTRA_METRIC_TASKS:
            rs = list(ev.rows.get((t.key, cond.final), {}).values())
            keys = sorted({k for r in rs for k in r if k.startswith(EXTRA_METRIC_PREFIXES)})
            extra.append({"task": t.key, **{k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) for k in keys}})
    out["log_loss_macro_f1"] = extra
    cost = []  # active cases (seed mean); timing lives in the run manifests' budgets
    for t in tasks:
        item = {"task": t.key}
        for name, c in (("F", cond.final), ("retrieval_only", cond.retrieval_only), ("default", cond.default)):
            vals = [r["n_cases"] for r in ev.rows.get((t.key, c), {}).values() if "n_cases" in r]
            if vals:
                item[name] = float(np.mean(vals))
        cost.append(item)
    out["active_cases"] = cost
    return out


def analyse(ev: Evidence, tasks: list[Task], cond: Conditions, prereg: dict, groups: list[str],
            min_var: float | None) -> dict:
    by_group = {g: [t for t in tasks if t.group == g] for g in groups}
    group_res = {}
    for g, ts in by_group.items():
        group_res[g] = {"tasks": [t.key for t in ts], "endpoints": endpoints(ev, ts, cond, prereg, min_var),
                        "guards": guards(ev, ts, cond, prereg, min_var)}
    primary = endpoints(ev, tasks, cond, prereg, min_var)
    guard = guards(ev, tasks, cond, prereg, min_var)
    transfer = group_res["transfer"]["endpoints"] if "transfer" in groups and len(groups) > 1 else None
    decide(primary, guard, transfer, prereg)
    # With one group the pooled scope is that group; listing it twice would double-count it in BH.
    sec = {"pooled": secondary(ev, tasks, cond),
           **({g: secondary(ev, ts, cond) for g, ts in by_group.items()} if len(by_group) > 1 else {})}
    tests = {f"{scope}/{name}": res[name]["p"] for scope, res in sec.items() for name in ("ece", "flips", "ranks")
             if math.isfinite(res[name].get("p", NAN))}
    adjusted = bh(tests)
    for key, q in adjusted.items():
        scope, name = key.split("/")
        sec[scope][name]["p_bh"] = q
        sec[scope][name]["bh_significant"] = q <= BH_Q
    return {"primary": {"groups": groups, "tasks": [t.key for t in tasks], "endpoints": primary, "guards": guard,
                        "transfer_check": None if transfer is None else {
                            k: {"gmr_ci95": transfer[k]["gmr_ci95"], "opposite": transfer[k]["gmr_ci95"][0] > 1.0}
                            for k in ("E1", "E2")}},
            "groups": group_res, "secondary": sec, "secondary_bh": {"q": BH_Q, "tests": tests, "adjusted": adjusted}}


# ----------------------------------------------------------------------------- output

def clean(x):
    """JSON-safe copy: NaN -> null, +-inf -> "inf"/"-inf"."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf") if math.isinf(x) else x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _f(x, fmt: str = ".3f") -> str:
    """Format a cleaned number: None/NaN -> 'n/a'; clean()'s "inf"/"-inf" strings pass through."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return x if isinstance(x, str) else format(x, fmt)


def _yes(x) -> str:
    return "n/a" if x is None else "yes" if x else "no"


def render_md(rep: dict) -> str:
    inp, prim = rep["inputs"], rep["primary"]
    short = lambda c: LABELS.get(c[9:], c[9:]) if c.startswith("baseline:") else c.split("::", 1)[-1]
    md = ["# T1 confirmation report (pre-registered)", "",
          f"Pre-registration `{inp['prereg']}` (sha256 `{rep['prereg_sha256'][:16]}...`, verified). "
          f"F = `{inp['final']}`; references: retrieval-only `{short(inp['retrieval_only'])}`, alternating default "
          f"`{short(inp['default'])}`, CatBoost and TabM HPO winners. Groups {' + '.join(prim['groups'])}, "
          f"{len(prim['tasks'])} tasks.", "",
          "Error: 1 - accuracy or train-standardized RMSE (test stream). GMR = exp(pooled mean log error ratio, "
          "eps = 0.5/n_test for classification); REML tau^2, Hartung-Knapp t(k-1). GMR < 1 favours F. "
          f"Holm over E1-E3b at {ALPHA}; secondary tests Benjamini-Hochberg at q = {BH_Q}.", ""]
    if rep["missing"]:
        md += [f"**Incomplete evidence** ({len(rep['missing'])} task/method pairs miss seeds); analysed with "
               "--allow-incomplete.", ""]
    md += ["## Primary endpoints", "",
           "| ID | F vs | tasks | GMR [95% CI] | one-sided 95% upper | p | p (Holm) | supported |", "|---|---|---|---|---|---|---|---|"]
    for k, r in prim["endpoints"].items():
        md.append(f"| {k} | {short(r['reference'])} | {r['k']} | {_f(r['gmr'])} [{_f(r['gmr_ci95'][0])}, "
                  f"{_f(r['gmr_ci95'][1])}] | {_f(r['gmr_upper95_one_sided'])} | {_f(r['p_one_sided'], '.4f')} | "
                  f"{_f(r['p_holm'], '.4f')} | {'yes' if r['supported'] else 'no'} |")
    md += [""] + [f"- **{k}**: {r['wording']}" + ("" if r["supported"] else " (failed: " + ", ".join(
        c for c, ok in r["criteria"].items() if not ok) + ")") for k, r in prim["endpoints"].items()]
    md += ["", "Decision rules: E1/E2 " + rep["decision_rules"]["superiority"] + "; E3a/E3b "
           + rep["decision_rules"]["noninferiority"] + "."]
    g = prim["guards"]
    md += ["", "## E2 guards", "", "| guard | value | limit | holds |", "|---|---|---|---|"]
    for name in ("pre_adaptation", "locality", "correction"):
        x = g[name]
        md.append(f"| {name} | {_f(x['value'])} | {x['limit']} | "
                  f"{'not recorded' if not x['evaluated'] else 'yes' if x['holds'] else 'no'} |")
    if prim["transfer_check"]:
        md += ["", "Transfer group check (no superiority claim if its interval lies above 1): " + "; ".join(
            f"{k} [{_f(v['gmr_ci95'][0])}, {_f(v['gmr_ci95'][1])}]{' opposite' if v['opposite'] else ''}"
            for k, v in prim["transfer_check"].items())]
    md += ["", "Non-inferiority sensitivity (descriptive): " + "; ".join(
        f"{k} delta {d}: {'bound below margin' if s['upper_bound_below_margin'] else 'not'}"
        for k in ("E3a", "E3b") for d, s in prim["endpoints"][k]["sensitivity"].items())]
    md += ["", "## By group (descriptive)", "", "| group | ID | tasks | GMR [95% CI] | p |", "|---|---|---|---|---|"]
    for grp, res in rep["groups"].items():
        for k, r in res["endpoints"].items():
            md.append(f"| {grp} | {k} | {r['k']} | {_f(r['gmr'])} [{_f(r['gmr_ci95'][0])}, {_f(r['gmr_ci95'][1])}] | "
                      f"{_f(r['p_one_sided'], '.4f')} |")
    md += ["", "## Secondary", "", "| scope | analysis | statistic | criterion holds | p | p (BH) |", "|---|---|---|---|---|---|"]
    for scope, res in rep["secondary"].items():
        e, fl, rk = res["ece"], res["flips"], res["ranks"]
        if e["available"]:
            md.append(f"| {scope} | ECE_T | median {_f(e['median_ece_post_T'], '.4f')} vs pre {_f(e['median_ece_pre'], '.4f')} "
                      f"+ {ECE_MARGIN} | {_yes(e['holds'])} | {_f(e['p'], '.4f')} | {_f(e.get('p_bh'), '.4f')} |")
        if fl["available"]:
            md.append(f"| {scope} | flips | {fl['wrong_to_correct']} / {fl['correct_to_wrong']} = {_f(fl['ratio'], '.2f')} | "
                      f"{'n/a (no flips)' if fl['holds'] is None else _yes(fl['holds'])} | {_f(fl['p'], '.4f')} | "
                      f"{_f(fl.get('p_bh'), '.4f')} |")
        md.append(f"| {scope} | ranks (Friedman) | {rk['tasks']} tasks | | {_f(rk['p'], '.4f')} | {_f(rk.get('p_bh'), '.4f')} |")
    names = list(next(iter(rep["secondary"].values()))["ranks"]["mean_rank"])
    md += ["", "Six-method mean ranks (1 = lowest error)", "", "| scope | " + " | ".join(names) + " |",
           "|---" * (len(names) + 1) + "|"]
    for scope, res in rep["secondary"].items():
        md.append(f"| {scope} | " + " | ".join(_f(res["ranks"]["mean_rank"][n], ".2f") for n in names) + " |")
    md += ["", "## Per task (F / reference error ratio, geometric mean over paired seeds)", "",
           "| task | group | seeds | F error | E1 | E2 | E3a | E3b | F pre / retrieval-only |", "|---|---|---|---|---|---|---|---|---|"]
    pt = {k: {p["task"]: p for p in r["per_task"]} for k, r in prim["endpoints"].items()}
    pre = {p["task"]: p for p in g["pre_adaptation"]["per_task"]}
    for t in rep["tasks"]:
        e1 = pt["E1"].get(t["key"], {})
        md.append(f"| {t['key']} | {t['group']} | {e1.get('n_seeds', 0)} | {_f(e1.get('error'), '.4f')} | "
                  + " | ".join(_f(pt[k].get(t["key"], {}).get("ratio")) for k in ("E1", "E2", "E3a", "E3b"))
                  + f" | {_f(pre.get(t['key'], {}).get('ratio'))} |")
    pv = rep["provenance"]
    md += ["", "## Provenance", "",
           f"- neural-CBR rows analysed {pv['rows_analysed']}, ignored {pv['rows_ignored']}, identical duplicates "
           f"{pv['duplicate_rows']}; run manifests found {pv['run_manifests_found']}, absent {pv['run_manifests_absent']}",
           f"- run source fingerprints: {', '.join(f'{k[:12]} ({sum(v.values())} rows)' for k, v in pv['source_fingerprints'].items()) or 'none recorded'}",
           f"- devices (rows): {pv.get('devices', {})}; comparisons pooling device types: "
           f"{len(pv.get('device_check', {}).get('mixed_comparisons', []))}",
           f"- n_test sources: {pv.get('n_test_sources', {})}"]
    md += [f"- warning: {w}" for w in rep["warnings"]]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--prereg-sha256", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="neural-CBR experiment dirs (runs__*.jsonl)")
    ap.add_argument("--baselines", nargs="*", default=[], help="grid or evaluate output dirs")
    ap.add_argument("--final", required=True, help="condition id of F, e.g. tuned::joint_summed_w5__...")
    ap.add_argument("--retrieval-only", help="default: <F's arm>::retrieval_only")
    ap.add_argument("--default", help="alternating default at F's budget; default: <F's arm>::alternating_default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=["legacy", "transfer"])
    ap.add_argument("--small-tasks", nargs="*", default=[], help="small-task names/keys, or a .json list")
    ap.add_argument("--n-test", help="fallback JSON {task key or name: n_test or {seed: n_test}}")
    ap.add_argument("--min-within-var", type=float, help="within-task variance floor (default: smallest positive)")
    ap.add_argument("--allow-mixed-code", action="store_true")
    ap.add_argument("--allow-mixed-device", action="store_true",
                    help="analyse E1/E2 comparisons whose neural-CBR arms ran on different device types")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    if args.min_within_var is not None and not args.min_within_var > 0:
        raise SystemExit("--min-within-var must be positive")
    prereg, digest = load_prereg(Path(args.prereg), args.prereg_sha256)
    tasks, unknown_small = plan_tasks(prereg, args.groups, small_task_names(args.small_tasks, prereg))
    arm = args.final.split("::", 1)[0] + "::" if "::" in args.final else ""
    cond = Conditions(args.final, args.retrieval_only or f"{arm}retrieval_only", args.default or f"{arm}alternating_default")
    if len({cond.final, cond.retrieval_only, cond.default}) != 3:
        raise SystemExit(f"F, retrieval-only and default must be distinct conditions: {cond}")
    ev = Evidence()
    if unknown_small:
        ev.warnings.append(f"small tasks not among the analysed tasks: {unknown_small}")
    load_runs(ev, [Path(d) for d in args.runs], tasks, cond, args.allow_mixed_code, args.allow_mixed_device)
    families = list(dict.fromkeys([prereg["endpoints"][e]["vs"] for e in ("E3a", "E3b")] + list(RANK_FAMILIES)))
    load_baselines(ev, [Path(d) for d in args.baselines], tasks, families)
    resolve_n_test(ev, tasks, json.loads(Path(args.n_test).read_text(encoding="utf-8")) if args.n_test else {})
    check_small_tasks(ev, tasks, args.allow_incomplete)
    required = [cond.final, cond.retrieval_only, cond.default, *endpoint_refs(prereg, cond).values()]
    missing = missing_evidence(ev, tasks, list(dict.fromkeys(required)))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"incomplete primary evidence for {len(missing)} task/method pair(s), e.g. {missing[:3]}; "
                         "pass --allow-incomplete to analyse what exists")
    report = {
        "prereg_sha256": digest, "document": prereg.get("document"),
        "inputs": {"prereg": args.prereg, "runs": args.runs, "baselines": args.baselines, "final": cond.final,
                   "retrieval_only": cond.retrieval_only, "default": cond.default, "groups": args.groups,
                   "small_tasks": sorted(t.key for t in tasks if t.small), "allow_mixed_code": args.allow_mixed_code,
                   "allow_mixed_device": args.allow_mixed_device, "allow_incomplete": args.allow_incomplete,
                   "min_within_var": args.min_within_var},
        "constants": {"alpha": ALPHA, "bh_q": BH_Q, "sensitivity_deltas": SENSITIVITY_DELTAS, "ece_margin": ECE_MARGIN,
                      "flip_factor": FLIP_FACTOR, "small_test_rows": SMALL_TEST_ROWS},
        "decision_rules": DECISION_RULES,
        "tasks": [{"key": t.key, "group": t.group, "small": t.small, "expected_seeds": list(t.seeds)} for t in tasks],
        "missing": missing,
        **analyse(ev, tasks, cond, prereg, args.groups, args.min_within_var),
        "provenance": ev.provenance, "warnings": ev.warnings,
    }
    report = clean(report)
    md = render_md(report)  # before writing anything, so a rendering error leaves no half-written report
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return report


if __name__ == "__main__":
    main()

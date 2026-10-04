"""Local common-knowledge reviewer demo with actual NN-kNN intervention states.

This is engineering preparation, not participant evidence. Displayed predictions,
case activations and C/H statistics share one frozen retrieval computation.
"""
from __future__ import annotations

import copy
import json
import math
import time
import zipfile
from pathlib import Path

import torch

from .core import CoreConfig, build_model, clone_optimizer, train_retrieval
from .maintenance import realign_optimizer_state
from .outcomes import final_prediction, frozen_evaluation
from .provenance import CaseStatisticsStore, ScoreConfig, score_cases, active_cohorts
from .revise import InterventionLog, flip_matrix
from .artifacts import source_fingerprint, git_state, environment_info


def common_knowledge_data():
    """Declared rule: red circles qualify; color/shape/size are observable."""
    X = torch.tensor([[c, s, size] for c in (0., 1.) for s in (0., 1.) for size in (.2, .8)])
    truth = ((X[:, 0] == 0) & (X[:, 1] == 0)).long()
    observed = truth.clone()
    observed[4:6] = 1  # explicitly injected blue-circle label errors
    queries = torch.tensor([[c, s, size] for c in (0., 1.) for s in (0., 1.) for size in (.35, .65)])
    target = ((queries[:, 0] == 0) & (queries[:, 1] == 0)).long()
    return X, observed, queries, target


class ReviewerSession:
    def __init__(self, output: Path, *, seed=11, epochs=30):
        if output.exists() and any(output.iterdir()):
            raise ValueError("use an empty output directory to preserve previous sessions")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        repo = Path(__file__).resolve().parents[2]
        fingerprint = source_fingerprint(repo)
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for directory in ("model", "tools", "configs", "tests", "datasets"):
                for path in sorted((repo / directory).rglob("*")):
                    if path.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in path.parts:
                        bundle.write(path, path.relative_to(repo).as_posix())
        (output / "session_manifest.json").write_text(json.dumps({
            "status": "engineering_demo_no_participants", "source_fingerprint": fingerprint,
            "git": git_state(repo), "environment": environment_info(), "seed": seed,
            "core_epochs_max": epochs, "injected_label_error_case_ids": [4, 5],
            "data_version": "common_red_circle/v1", "schema": "reviewer_ui/v1",
            "validation_sizes": [.1, .9], "display_query_sizes": [.35, .65],
            "training_sizes": [.2, .8], "rule": "red and circle, size irrelevant",
            "retrieval_event_semantics": "same event for prediction, display and C/H/A",
            "statistic_reference": "eight fixed demo queries; no broad reliability inference"}, indent=2), encoding="utf-8")
        self.started = time.monotonic()
        self.X, self.targets, self.queries, self.query_targets = common_knowledge_data()
        self.cfg = CoreConfig(task_type="classification", seed=seed, epochs=epochs,
                              patience=epochs, batch_size=8, top_k=3, bias_init_k=3)
        self.model = build_model(self.X, self.targets, self.cfg, 2)
        self.validation = self.queries.clone()
        self.validation[:, 2] = torch.tensor([.1, .9] * 4)
        trained = train_retrieval(self.model, self.X, self.targets, self.validation, self.query_targets, self.cfg)
        self.optimizer = trained.optimizer
        self.m0 = copy.deepcopy(self.model)
        self.m0_optimizer = clone_optimizer(self.optimizer, self.m0, self.cfg)
        self.m0_targets = self.targets.clone()
        self.store = CaseStatisticsStore()
        self.log = InterventionLog("common-knowledge-demo")
        self.version = 0
        self.actions = []
        self.controlled_result = None
        self.retraining = None
        self.snapshot = self._view()
        self.baseline_predictions = torch.tensor([q["prediction"] for q in self.snapshot["queries"]])
        self._save("M0")

    @staticmethod
    def object_description(row):
        return {"color": "红色" if row[0] == 0 else "蓝色",
                "shape": "圆形" if row[1] == 0 else "方形", "size": float(row[2])}

    def _view(self):
        self.model.eval()
        ids = self.model.active_case_ids()
        self.store.reset_counts()
        with frozen_evaluation(self.model):
            r = self.model.retrieve(self.queries, exclude_identical=False, case_mask=self.log.case_mask(self.model))
            pre, final, _ = final_prediction(self.model, self.queries, r)
            w = r["weights"]
            correct = final.argmax(1) == self.query_targets
            for j, slot in enumerate(r["case_indices"]):
                st = self.store.ensure(int(ids[slot]))
                st.retrieval_count = float((w[:, j] > 0).sum())
                st.activation_mass = float(w[:, j].sum())
                st.positive_support = float(w[correct, j].sum())
                st.harmful_support = float(w[~correct, j].sum())
                st.error_support = st.harmful_support
                st.audits = 1
            scores = score_cases(ids.numpy(), active_cohorts(self.model),
                                 self.model.biases[:self.model.case_count()].detach().numpy(),
                                 self.store, ScoreConfig(1., 1., .01))
            cases = []
            for slot, cid in enumerate(ids.tolist()):
                cases.append({"id": cid, **self.object_description(self.model.cases[slot]),
                              "label": int(self.model.labels[slot].argmax()), **scores.row(slot),
                              "quarantined": cid in self.log.quarantined, "protected": self.store.get(cid).protected})
            events = []
            for i in range(len(self.queries)):
                selected = w[i] > 0
                slots = r["case_indices"][selected]
                events.append({"id": i, "retrieval_event_id": f"ui-v{self.version}-q{i}",
                               **self.object_description(self.queries[i]), "prediction": int(final[i].argmax()),
                               "probabilities": final[i].tolist(), "case_ids": ids[slots].tolist(),
                               "activations": w[i, selected].tolist(), "distances": r["distances"][i, selected].tolist(),
                               "biases": self.model.biases[slots].tolist()})
        return {"version": self.version, "rule": "红色且为圆形的物体符合条件；大小无关。",
                "cases": cases, "queries": events, "archive_ids": sorted(self.log.archive.entries),
                "feature_weights_raw": self.model.glocal_weightor.feature_weights.detach().tolist(),
                "feature_weights_projected": torch.nn.functional.leaky_relu(
                    self.model.glocal_weightor.feature_weights.detach(), negative_slope=.001).tolist(),
                "case_weights_raw": self.model.glocal_weights[:self.model.case_count()].detach().tolist(),
                "case_feature_weight_factors": (torch.nn.functional.leaky_relu(
                    self.model.glocal_weights[:self.model.case_count()].detach(), negative_slope=.001)
                    @ torch.nn.functional.leaky_relu(self.model.glocal_weightor.feature_weights.detach(), negative_slope=.001)).tolist(),
                "actions": self.actions, "controlled_result": self.controlled_result,
                "retraining": self.retraining, "study_status": "工程演示，尚未开展参与者研究"}

    def _save(self, stage):
        folder = self.output / f"v{self.version:04d}-{stage}"
        folder.mkdir()
        torch.save({"model_state": self.model.state_dict(), "active_case_count": self.model.case_count(),
                    "optimizer_state": self.optimizer.state_dict(), "interventions": self.log.state_dict(),
                    "statistics": self.store.state_dict(), "training_targets": self.targets,
                    "X_train": self.X, "X_queries": self.queries, "y_queries": self.query_targets,
                    "core": self.cfg.__dict__}, folder / "checkpoint.pt")
        (folder / "view.json").write_text(json.dumps(self.snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output / "latest.json").write_text(json.dumps({"version": self.version, "stage": stage,
            "folder": folder.name}, indent=2), encoding="utf-8")

    def act(self, request):
        if self.retraining is not None:
            raise ValueError("本次 M2／MC 比较已完成；请使用新的演示会话继续修改")
        if type(request.get("version")) is not int or request.get("version") != self.version:
            raise ValueError("界面已更新，请刷新后重试")
        for name in ("actor", "reason", "expected_effect"):
            if not isinstance(request.get(name), str) or not request[name].strip():
                raise ValueError("请填写审阅者、理由和预期效果")
        if request["expected_effect"] not in {"improve", "unchanged", "uncertain"}:
            raise ValueError("invalid expected effect")
        confidence = request.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        qid = request.get("query_id")
        if type(qid) is not int or not 0 <= qid < len(self.queries):
            raise ValueError("select a valid target query")
        op, cid = request.get("operation"), request.get("case_id")
        if op not in {"set_feature_weights", "force", "undo", "retrain"} and (type(cid) is not int or not 0 <= cid < len(self.X)):
            raise ValueError("select a valid stable case ID")
        # The tuple copy preserves optimizer-to-parameter references during rollback.
        saved = copy.deepcopy((self.model, self.optimizer, self.store, self.log, self.targets,
                               self.version, self.actions, self.snapshot, self.controlled_result, self.retraining))
        before = torch.tensor([q["prediction"] for q in self.snapshot["queries"]])
        prior_event = copy.deepcopy(self.snapshot["queries"][qid])
        kwargs = {"actor": request["actor"], "reason": request["reason"]}
        try:
            if op == "relabel":
                self.log.relabel(self.model, cid, request["label"], **kwargs)
                self.targets[cid] = int(request["label"])
            elif op == "adjust_bias":
                self.log.adjust_bias(self.model, cid, float(request["delta"]), **kwargs)
            elif op in {"quarantine", "release", "protect", "unprotect"}:
                method = getattr(self.log, op)
                args = (self.model, cid, self.store) if op in {"protect", "unprotect"} else (self.model, cid)
                method(*args, **kwargs)
            elif op in {"archive_remove", "restore"}:
                args = (self.model, cid, self.store) if op == "archive_remove" else (self.model, cid)
                getattr(self.log, op)(*args, optimizer=self.optimizer, **kwargs)
            elif op in {"set_feature_weights", "set_case_weights"}:
                method = getattr(self.log, op)
                args = (self.model, request["weights"]) if op == "set_feature_weights" else (self.model, cid, request["weights"])
                method(*args, **kwargs)
            elif op == "undo":
                iid = request["intervention_id"]
                record = next(r for r in self.log.records if r["intervention_id"] == iid)
                self.log.undo_parameter_edit(self.model, iid, **kwargs)
                if record["operation"] == "relabel":
                    self.targets[record["case_id"]] = record["before"]
            elif op == "force":
                _, prediction, _ = self.log.controlled_decision(self.model, self.queries[qid:qid+1],
                    request["case_ids"], query_ids=[qid], activation_floor=float(request["activation_floor"]), **kwargs)
                self.controlled_result = {"query_id": qid, "prediction": int(prediction.argmax(1)[0]),
                                          "event": copy.deepcopy(self.log.decisions[-1]), "persistent": False}
            elif op == "retrain":
                self._retrain(request["epochs"])
            else:
                raise ValueError("unsupported reviewer operation")
            self.version += 1
            self.snapshot = self._view()
            after = torch.tensor([q["prediction"] for q in self.snapshot["queries"]])
            effect_after = after.clone()
            if op == "force":
                effect_after[qid] = self.controlled_result["prediction"]
            affected = torch.tensor([cid in q["case_ids"] for q in saved[7]["queries"]]) if cid is not None else torch.ones(len(before), dtype=torch.bool)
            action = {"version": self.version, "operation": op, "case_id": cid, **kwargs,
                "expected_effect": request["expected_effect"], "confidence": confidence,
                "target_query_id": qid, "predeclared_target_event": prior_event,
                "elapsed_seconds": time.monotonic() - self.started,
                "flips": flip_matrix(before, after, self.query_targets),
                "target_decision_before": int(before[qid]), "target_decision_after": int(effect_after[qid]),
                "target_was_correct": bool(before[qid] == self.query_targets[qid]),
                "target_is_correct": bool(effect_after[qid] == self.query_targets[qid]),
                "targeted_flips": flip_matrix(before[affected], after[affected], self.query_targets[affected]),
                "collateral_flips": flip_matrix(before[~affected], after[~affected], self.query_targets[~affected])}
            self.actions.append(action)
            self.snapshot["interventions"] = copy.deepcopy(self.log.records)
            self._save("M2" if op == "retrain" else "M1")
            return self.snapshot
        except Exception:
            (self.model, self.optimizer, self.store, self.log, self.targets,
             self.version, self.actions, self.snapshot, self.controlled_result, self.retraining) = saved
            raise

    def _retrain(self, epochs):
        if type(epochs) is not int or not 1 <= epochs <= 20:
            raise ValueError("controlled retraining requires 1 to 20 explicit epochs")
        if self.retraining is not None:
            raise ValueError("this demo permits one M2/MC comparison per session")
        model = copy.deepcopy(self.model)
        opt = clone_optimizer(self.optimizer, model, self.cfg)
        mask = self.log.case_mask(model)
        if mask is not None:
            old = model.case_count()
            keep = torch.nonzero(mask).view(-1)
            if not len(keep):
                raise ValueError("no eligible cases remain")
            model.compact_cases(keep)
            realign_optimizer_state(opt, model, keep.numpy(), old)
        control = copy.deepcopy(self.m0)
        copt = clone_optimizer(self.m0_optimizer, control, self.cfg)
        cfg = copy.copy(self.cfg)
        cfg.patience = epochs
        with torch.random.fork_rng():
            result = train_retrieval(model, self.X, self.targets, self.validation, self.query_targets,
                                    cfg, epochs=epochs, optimizer=opt, select_best=False, lr_scale=.1)
        with torch.random.fork_rng():
            ctr = train_retrieval(control, self.X, self.m0_targets, self.validation, self.query_targets,
                                 cfg, epochs=epochs, optimizer=copt, select_best=False, lr_scale=.1)
        with frozen_evaluation(control):
            r = control.retrieve(self.queries, exclude_identical=False)
            _, prediction, _ = final_prediction(control, self.queries, r)
        torch.save({"model_state": control.state_dict(), "active_case_count": control.case_count(),
                    "optimizer_state": ctr.optimizer.state_dict(), "history": ctr.history,
                    "training_targets": self.m0_targets}, self.output / "MC.pt")
        self.model, self.optimizer = model, result.optimizer
        self.retraining = {"epochs": epochs, "lr_scale": .1, "selection": "fixed final epoch",
                           "M2_history": result.history, "MC_history": ctr.history,
                           "MC_predictions": prediction.argmax(1).tolist(),
                           "capacity_M2": model.case_count(), "capacity_MC": control.case_count(),
                           "budget_note": "same examples/epochs; compaction can change retrieval cost"}

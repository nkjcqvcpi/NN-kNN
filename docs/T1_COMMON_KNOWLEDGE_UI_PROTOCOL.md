# Common-knowledge reviewer UI protocol — engineering draft, 2026-10-04

This prepares the required T1.2 interface study. It is not a completed study,
a PI-approved protocol or authorization to recruit participants. Implementation
can build and test the interface independently of human data collection.
Simulated engineering reviewers remain distinctly labelled.

## Proposed task and interface

Use ordinary fruit/vehicle categorization with short named objects, declared
observable features and objectively checkable reference categories. Exclude
healthcare, personal information, specialized judgments and consequential
actions. Freeze category/feature definitions, injected corruption, task order,
query assignments and reference answers before recruitment. Measure ceiling
effects: a successful correction may have no test effect when its case is
rarely retrieved.

Show the query, current decision, actual retrieved case names/solutions,
learned distances, normalized activation and biases. Show evidence sufficiency
and distinguish a stored solution from the final query outcome credited to it.
Present raw feature parameters separately from projected distance weights.
Provide controls to correct a category, change case bias/mixture weights or
feature parameters, force an eligible case for one decision, protect,
quarantine/release and archive/restore a reviewed case.

Record each action's intended effect and reason, apply the real reviewer API,
then display immediate predictions from the same logged event. Mark force
overrides visibly, including activation floor and scope. Offer rollback and
explain when intervening optimization requires full checkpoint restoration.
Controlled retraining is a separate action with an explicit epoch/rate budget.
Show M0, M1, M2 and the research-only matched no-edit continuation separately.
Preserve predictions and edited case states for replay.

## Measures and comparison

Before editing, ask the participant to identify the suspect case and predict
its effect. Record correct diagnosis, intended and actual decision changes,
beneficial/harmful flips, collateral changes, post-training persistence and
agreement between displayed influence and the intervention outcome. Record
active review time, action count/types, reversals, unsuccessful attempts,
completion and participant confidence/burden. Include attempts and exclusions.

Compare complete versus prespecified reduced evidence under matched tasks and
model snapshots. Counterbalance order and avoid revealing a task's reference
answer in one condition before testing it in another. Freeze participant count,
assignment, margins, analysis, uncertainty and stopping rules with the PI
before collecting participant data, rather than after seeing results.

## Engineering acceptance

Load a trained baseline, apply actual API changes, save stage checkpoints/logs,
replay predictions and recover original behavior after rollback. Check eligibility
gates, invalid inputs, capacity conflicts and optimizer alignment. Verify that
timestamps, intended effects and outcomes persist; use neutral task/error
messages and clear state changes. Rehearsals use a simulated-reviewer ID and
cannot enter participant tables. Institutional review and actual participants
remain prerequisites for a human-study claim.

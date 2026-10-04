# Synchronization checkpoint selection repair

The regression/classification synchronization loop previously discarded its best
checkpoint at every maintenance hook, even when the hook kept every case and
changed no state. It also delayed early stopping for future maintenance points
in schedules that never execute maintenance. Three focused tests reproduced the
first two failures and verify that a real label edit at unchanged capacity still
resets selection. The repaired suite passes 78 tests.

The loop now compares actual model state and active capacity across maintenance,
records `maintenance_state_changed`, and resets selection only after a real
change. Only RRR with an executable future hook delays stopping. Radius
calibration remains an explicit scheduled operation and travels with the chosen
checkpoint; a calibration snapshot step is not a serialized best epoch.

`alignment_sync_selection_fix.yaml` repeats the previous 36-run exploratory
pilot: four datasets, seeds 5/6/7, independent/RR/RRR. Frozen source SHA256 is
`b39dad2a7a2dfab627abe7a57a609902f4a7892a7107e065d147c142f2231890`.
All 36 checkpoint/adapter predictions and source bundles replay. All 12 RRR
epoch-8 hooks compress and record true state changes; all epoch-16 hooks keep
original slots and record no state change. Independent/RR metrics and event
predictions agree with the old batch within 1e-7.

Synthetic seed 7 RRR post accuracy rises from 0.919231 to 0.934615. Other reported
post metrics remain unchanged. Energy RRR-minus-independent RMSE remains
+0.014386/+0.006908/+0.007966. This repair therefore does not explain the energy
compression loss or resolve unequal capacity between RRR and the other arms.
Raw runs: `results/t1_pi20260920/continuation_20261004/alignment_sync_selection_fix`.
Chat outputs contain the full paired report and independent replay JSON. This
is a bounded diagnostic repeat, not a confirmatory comparison.

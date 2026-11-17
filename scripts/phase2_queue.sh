#!/bin/zsh
# Full Phase 2 methodology run (restarted 2026-11-15 after the synthetic file-name collision fix in
# src/train/trainer.py: every SEQUENCE model is re-trained; tabular B0-B2 were not affected).
# Three chains run in parallel on the M4; each step logs to results/runs/queue_<script>.log and
# one START/END line (with the real exit code) to results/runs/queue.log.
cd "${0:A:h}/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
R=results/runs
run() {
  echo "$(date '+%F %T') START $*" >> $R/queue.log
  "$@" >> $R/queue_${2:t:r}.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END($rc) $*" >> $R/queue.log
}

chain_a() {   # E1 sequence baselines, B9, robustness tests that need the PIPS ensemble
  run python experiments/e1_sequence.py --models B3 B4 B5 B6 B7 B8 B8i --tag main
  run python experiments/b9_lstm_ae.py
  run python experiments/e1_sequence.py --split v1v2 --models B3 B4 B5 B6 B7 B8 B8i --tag v1v2
  run python experiments/b9_lstm_ae.py --split v1v2
}
chain_b() {   # PI-PatchSurv: 5-seed ensemble, E2 ablations, +SI ensemble, v1 -> v2
  run python experiments/e1_sequence.py --models PIPS --seeds 0 1 2 3 4 --tag main_pips
  run python experiments/e1_sequence.py --models PIPS-phys PIPS-mono PIPS-surv PIPS-patch PIPS-nosyn --tag e2
  run python experiments/e1_sequence.py --models PIPS-SI --seeds 0 1 2 3 4 --tag si
  run python experiments/e1_sequence.py --split v1v2 --models PIPS --seeds 0 1 2 3 4 --tag v1v2_pips
}
chain_c() {   # tabular + E1 Optuna (equal budget), independent of the untuned runs
  run python experiments/e1_baselines.py --models B2-SI
  run python experiments/e1_tune.py --models B1 B2 B3 B4 B5 B6 B7 B8 B8i B9 PIPS --trials 20
}
chain_a & chain_b & chain_c &
wait
# needs the PIPS / B3 checkpoints above
run python experiments/e3_robustness.py --models PIPS B3
run python experiments/e3_resampling.py --models PIPS B3
# re-train with the tuned parameters
run python experiments/e1_baselines.py --models B1-tuned B2-tuned
run python experiments/b9_lstm_ae.py --tuned
run python experiments/e1_sequence.py --models B3-tuned B4-tuned B5-tuned B6-tuned B7-tuned B8-tuned B8i-tuned --tag tuned
run python experiments/e1_sequence.py --models PIPS-tuned --seeds 0 1 2 3 4 --tag tuned_pips
run python experiments/analyze.py
run python experiments/analyze.py --split v1v2
echo "$(date '+%F %T') QUEUE DONE" >> $R/queue.log

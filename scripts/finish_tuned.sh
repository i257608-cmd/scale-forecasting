#!/bin/zsh
# Finish E1-tuned after the Kaggle B5/B6 studies arrive (2026-11-16). The queue's tuned B3-B8i step
# stopped at B5 (no Optuna result yet, tuned_params raises). Waits for the Kaggle JSONs and for the
# main queue to finish (one MPS job at a time: 3 parallel jobs segfaulted on 2026-11-15), then
# re-runs that step with --resume (keeps finished runs) and the final analysis.
cd "${0:A:h}/.."
K=$1   # folder holding the downloaded Kaggle output (results/tuning/B5.json, B6.json)
log() { echo "$(date '+%F %T') $*" >> results/runs/queue.log; }
until [[ -f $K/results/tuning/B5.json && -f $K/results/tuning/B6.json ]]; do sleep 300; done
for m in B5 B6; do
  python3 -c "import json,sys; sys.exit(len(json.load(open('$K/results/tuning/$m.json')))<6)" \
    || { log "finish_tuned: $m.json has < 6 folds, stopping"; exit 1; }
  cp -n $K/results/tuning/$m.json results/tuning/
done
until grep -q "QUEUE DONE" results/runs/queue.log; do sleep 300; done
source .venv/bin/activate
for cmd in "python experiments/e1_sequence.py --models B3-tuned B4-tuned B5-tuned B6-tuned B7-tuned B8-tuned B8i-tuned --tag tuned --resume" \
           "python experiments/analyze.py" "python experiments/analyze.py --split v1v2"; do
  log "START $cmd"
  eval $cmd >> results/runs/queue_finish_tuned.log 2>&1
  log "END($?) $cmd"
done
log "FINISH_TUNED DONE"

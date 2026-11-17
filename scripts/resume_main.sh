#!/bin/zsh
# Resume E1 B3-B8i LOWO after the 2026-11-15 23:24 segfault (exit 139, MPS, 3 jobs in parallel).
# Waits for chain_a of phase2_queue.sh to finish (less GPU contention), then retries up to 3x;
# --resume skips (model, well, seed) runs already done after the key fix.
cd "${0:A:h}/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
R=results/runs
until grep -q "END.*b9_lstm_ae.py --split v1v2" $R/queue.log; do sleep 120; done
for i in 1 2 3; do
  echo "$(date '+%F %T') START resume_main try $i" >> $R/queue.log
  python experiments/e1_sequence.py --models B3 B4 B5 B6 B7 B8 B8i --tag main_resume --resume >> $R/queue_resume_main.log 2>&1
  rc=$?
  echo "$(date '+%F %T') END($rc) resume_main try $i" >> $R/queue.log
  [[ $rc == 0 ]] && break
done

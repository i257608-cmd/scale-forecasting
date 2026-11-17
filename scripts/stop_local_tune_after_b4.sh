#!/bin/zsh
# B5-B9 and PIPS are tuned on Kaggle (2 x T4, 2026-11-16). Stop the local e1_tune.py once
# B4 has all 6 folds so it does not start B5. Results of B1-B4 stay in results/tuning/.
cd "${0:A:h}/.."
until [[ -f results/tuning/B4.json ]] && python3 -c "import json,sys; sys.exit(len(json.load(open('results/tuning/B4.json')))<6)"; do sleep 120; done
pkill -f "experiments/e1_tune.py" && echo "$(date '+%F %T') stopped local e1_tune after B4 (B5-PIPS on Kaggle)" >> results/runs/queue.log

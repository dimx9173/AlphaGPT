#!/usr/bin/env bash
# Reduced-grammar real-vs-null batch.
#
# Matched to results/ga_iter_runs and results/ga_null_runs in every respect that
# is not the grammar: same seeds, same generations, same population, same
# reward version, same real funding basis, same iid null construction with the
# same null seed. The only difference is --grammar reduced. That is what makes
# the two null batches comparable: if the reduced grammar still manufactures
# lockbox Sharpe on structureless data, the problem is not search freedom, and
# no amount of further grammar reduction will be worth doing.
set -u
cd /home/brian/project/AlphaGPT
PY=.venv2/bin/python
SEEDS="${SEEDS:-101 102 103 104 105 106 107 108 109 110}"
GENS="${GENS:-100}"
POP="${POP:-32}"
PAR="${PAR:-5}"
GRAMMAR="${GRAMMAR:-reduced}"
OUT_ROOT="${OUT_ROOT:-results/gram_${GRAMMAR}_runs}"
mkdir -p "$OUT_ROOT" logs

run_one () {
  local kind="$1" seed="$2" extra="$3"
  local out="$OUT_ROOT/${kind}_seed${seed}.json"
  local log="logs/gram_${GRAMMAR}_${kind}_seed${seed}.log"
  if [ -s "$out" ]; then echo "skip $out"; return 0; fi
  $PY research/ga_28c_30m_3y.py \
      --seed "$seed" --generations "$GENS" --population "$POP" \
      --grammar "$GRAMMAR" --funding real --reward-version v1 $extra \
      --out "$out" > "$log" 2>&1
  echo "done ${kind} seed${seed} exit=$?"
}

i=0
for seed in $SEEDS; do
  run_one real "$seed" "" &
  run_one null "$seed" "--null iid --null-seed 0" &
  i=$((i+2))
  if [ $((i % PAR)) -eq 0 ]; then wait; fi
done
wait
echo "BATCH COMPLETE grammar=$GRAMMAR"

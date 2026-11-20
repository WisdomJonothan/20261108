#!/bin/bash
cd ~/RETECO
M=/tmp/reteco/ce_ft
for F in f0 f1 f2 fall; do
  while [ ! -f $M/$F.trained ]; do sleep 30; done
  [ -f $M/$F.inferred ] && continue
  echo "[infer] start $F $(date)"
  if GPU=2 bash gate_ce_infer.sh $F > logs/${F}_infer.log 2>&1; then
    touch $M/$F.inferred; echo "[infer] ok $F $(date)"
    rm -rf $M/$F; echo "[infer] freed checkpoint $F"
  else
    echo "[infer] FAIL $F $(date)"; tail -5 logs/${F}_infer.log
  fi
done
echo "[infer] ALL_DONE $(date)"

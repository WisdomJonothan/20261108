import json
import subprocess

BASH = r"C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"
cmd = (
    "cd /d/RETECO && PYTHONUNBUFFERED=1 PYTHONUTF8=1 "
    "/d/anaconda/python.exe scripts/train_ce.py "
    "--model models/bge-reranker-v2-m3 "
    "--domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace "
    "--groups-prefix results/ce_train_groups --run-tag mxbai "
    "--val-frac 0.12 --val-queries 4 --val-depth 100 "
    "--freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 "
    "--epochs 2 --lr 1.5e-5 --seed 29 --tag v2m3_pool8_e2s29 --outdir results/ce_ft "
    "> logs/detached_e2s29b.out 2>&1"
)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

p = subprocess.Popen(
    [BASH, "-lc", cmd],
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
    close_fds=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print("launched pid", p.pid)
json.dump({"pid": p.pid, "cmd": cmd}, open("logs/detached_e2s29b.json", "w"))

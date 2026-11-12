"""After the seed-29 arm is frozen, run the synthetic test-split replay.

Waits for results/aggregate_sub_e2s29_dev.json (written at the end of
phase12_post.sh), then runs scripts/phase13_test_replay.sh for three small
domains.  Launched detached so harness job kills cannot reach it.
"""
import os
import subprocess
import time

ROOT = r"D:\RETECO"
FLAG = os.path.join(ROOT, "results", "aggregate_sub_e2s29_dev.json")
LOG = os.path.join(ROOT, "logs", "phase13_test_replay.log")
BASH = r"C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"

deadline = time.time() + 150 * 60
while time.time() < deadline:
    if os.path.isfile(FLAG):
        break
    time.sleep(60)
else:
    open(LOG, "a").write("[watch13] timeout waiting for frozen seed-29 arm\n")
    raise SystemExit(2)

with open(LOG, "a") as f:
    f.write(f"[watch13] start at {time.strftime('%H:%M:%S')}\n")
    r = subprocess.run(
        [BASH, "-lc",
         "cd /d/RETECO && CKPT=results/ce_ft/v2m3_pool8_e2/best OUT=results/test_replay "
         "bash scripts/phase13_test_replay.sh iota quant law"],
        stdout=f, stderr=subprocess.STDOUT)
with open(LOG, "a") as f:
    f.write(f"[watch13] exit={r.returncode} at {time.strftime('%H:%M:%S')}\n")

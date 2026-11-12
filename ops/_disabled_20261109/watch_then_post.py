"""Wait for the seed-29 checkpoint, then run the post-training chain detached.

Launch this once with Start-Process (or a detached Popen); it survives harness
job kills because it is not in the job tree.
"""
import os
import subprocess
import sys
import time

ROOT = r"D:\RETECO"
RUN = os.path.join(ROOT, "results", "ce_ft", "v2m3_pool8_e2s29")
LOG = os.path.join(ROOT, "logs", "phase12_post_e2s29.log")
BASH = r"C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"

deadline = time.time() + 90 * 60
while time.time() < deadline:
    if os.path.isdir(os.path.join(RUN, "best")):
        break
    time.sleep(30)
else:
    open(LOG, "a").write("[watch] timeout waiting for best/ checkpoint\n")
    sys.exit(2)

with open(LOG, "a") as f:
    f.write(f"[watch] checkpoint found at {time.strftime('%H:%M:%S')}; starting post chain\n")
env = dict(os.environ, SUFFIX="e2s29")
with open(LOG, "a") as f:
    r = subprocess.run([BASH, "-lc", "cd /d/RETECO && SUFFIX=e2s29 bash scripts/phase12_post.sh"],
                       stdout=f, stderr=subprocess.STDOUT, env=env)
f2 = open(LOG, "a")
f2.write(f"[watch] post chain exit={r.returncode} at {time.strftime('%H:%M:%S')}\n")
f2.close()

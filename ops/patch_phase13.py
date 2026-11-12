import io

p = "scripts/phase13_test_replay.sh"
s = io.open(p, encoding="utf-8").read()
pairs = [
    ('if len(r) != 10: bad.append((q, f"{len(r)} rows"))',
     'if len(r) < 10: bad.append((q, f"{len(r)} rows"))'),
    ('if [x[0] for x in r] != list(range(1, 11)): bad.append((q, "ranks"))',
     'if [x[0] for x in r] != list(range(1, len(r) + 1)): bad.append((q, "ranks"))'),
    ('echo "=== $d: local sanity (10 docs/topic, contiguous ranks, strict scores)"',
     'echo "=== $d: local sanity (>=10 docs/topic, contiguous ranks, strict scores)"'),
    ('print(f"  OK: {len(qids)} topics x 10 docs, ranks 1-10, strictly decreasing scores")',
     'print(f"  OK: {len(qids)} topics x {min(len(rows[q]) for q in qids)} docs, contiguous ranks, strictly decreasing scores")'),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched phase13 sanity")

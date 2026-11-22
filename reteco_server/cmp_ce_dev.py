"""Compare two CE dev evaluations (metric-aligned vs frozen checkpoint) domain by domain."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--a", default="frozenrepro", help="baseline tag under results/aggregate_<tag>_dev.json")
p.add_argument("--b", default="ndcgdev", help="candidate tag")
a = p.parse_args()

root = Path(__file__).resolve().parent.parent

def load(tag):
    path = root / "results" / f"aggregate_{tag}_dev.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

A, B = load(a.a), load(a.b)
if A is None or B is None:
    raise SystemExit(f"missing aggregate: {a.a}={A is not None} {a.b}={B is not None}")

print(f"{'domain':12s} {a.a:>10s} {a.b:>10s} {'delta':>9s}")
deltas = []
for dom in sorted(A["per_domain"]):
    x = A["per_domain"][dom]["nDCG@10"]
    y = B["per_domain"][dom]["nDCG@10"]
    deltas.append(y - x)
    print(f"{dom:12s} {x:10.4f} {y:10.4f} {y - x:+9.4f}")
print(f"{'MACRO':12s} {A['domain_macro']:10.4f} {B['domain_macro']:10.4f} {B['domain_macro'] - A['domain_macro']:+9.4f}")
print(f"{'topic':12s} {A.get('topic_macro', float('nan')):10.4f} {B.get('topic_macro', float('nan')):10.4f} {'':>9s}")
print(f"win {sum(d > 0 for d in deltas)}/13  mean delta {sum(deltas)/len(deltas):+.4f}")

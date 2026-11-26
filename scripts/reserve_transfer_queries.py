"""Reserve a new domain-stratified training confirmation sample by ID alone."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json

from compare_anchor_reference import load


SALT = "held-transfer-v1|2026-11-26"


def key(row):
    return row["domain"], row["qid"]


def select(source, used, per_domain):
    stored = set()
    domains = defaultdict(list)
    for row in source:
        identity = key(row)
        if identity in stored or row["split"] != "train":
            raise ValueError("Expected unique official training question IDs")
        stored.add(identity)
        if identity not in used:
            domains[row["domain"]].append(row)
    if not used <= stored:
        raise ValueError("The reserve source does not contain all excluded training IDs")

    def order(row):
        domain, qid = key(row)
        return sha256(f"{SALT}|{domain}|{qid}".encode()).digest()

    return [row for domain, rows in sorted(domains.items())
            for row in sorted(rows, key=order)[:per_domain]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "excluded-manifest", "excluded-ids", "ids", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=30)
    args = parser.parse_args()
    if args.per_domain < 1 or any(p.exists() for p in (args.ids, args.manifest)):
        parser.error("Use fresh outputs and a positive domain cap")
    old = {key(row) for row in load(args.excluded_manifest)["queries"]}
    confirmation = {(d,q) for d,q,_ in load(args.excluded_ids)}
    if old & confirmation or len(old) != 186 or len(confirmation) != 155:
        raise ValueError("Unexpected preceding samples")
    pool = select(load(args.source)["queries"], old | confirmation, args.per_domain)
    counts = defaultdict(int)
    for row in pool:
        counts[row["domain"]] += 1
    ids = {"salt": SALT, "per_domain_cap": args.per_domain, "per_domain": dict(counts),
           "excluded": len(old | confirmation), "queries": [key(row) for row in pool]}
    for path, result in ((args.ids, ids), (args.manifest, {"queries": pool})):
        with path.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    print(f"Reserved {len(pool)} training questions in {len(counts)} domains", dict(counts))


if __name__ == "__main__":
    main()

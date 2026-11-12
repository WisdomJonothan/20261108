import io

path = r"D:\RETECO\github_repo\README.md"
lines = io.open(path, encoding="utf-8", newline="").read().split("\n")

start = next(i for i, l in enumerate(lines)
             if l.startswith("Result under the official metric, at the train-selected weight 1:1:"))
end = next(i for i, l in enumerate(lines)
           if l.startswith("See `results/PHASE9_PLAN.md` and `docs/RESEARCH_LOG.md`"))
assert start < end

new_block = """**Result under the official metric.** Two schedules were run; the longer one is
the frozen system and the shorter one is the ablation:

| dev (519 topics) | domain-macro | topic-macro |
|---|---:|---:|
| dense + zero-shot CE, z 1:1 | 0.3249 | 0.3260 |
| dense + SFT CE, z 1:1 (1 epoch, lr 1e-5) | 0.3336 | 0.3339 |
| **dense + SFT CE, z 1:1 (2 epochs, lr 1.5e-5)** | **0.3581** | **0.3644** |

The 2-epoch arm vs zero-shot at the same frozen weight: **+0.0332**,
query-bootstrap CI [+0.0161,+0.0523], domain-clustered [+0.0192,+0.0472],
188 queries better / 78 worse / 253 tied, 11 of 13 domains up. The 1-epoch arm
gives **+0.0087** [+0.0025,+0.0151] / domain-clustered [+0.0037,+0.0142]; it is
the ablation that isolates "more training" from "any training". The total over
V1 for the 1-epoch arm is **+0.0170** [+0.0055,+0.0290], domain-clustered
[+0.0052,+0.0287], which bundles the CE swap with a fusion-form change worth
+0.0083 whose own CI crosses zero — the 2-epoch arm is therefore reported as a
single-factor claim against the zero-shot fusion, not against V1.

The pre-registered adoption bar (+0.015 on dev **topic-macro**) was **not** met
by the 1-epoch arm: under the registered statistic SFT − V1 = 0.3339 − 0.3199 =
+0.0141. The bar is not restated under the corrected metric; the significance
claims above are made with the official domain-macro bootstrap instead. Frozen
as `runs/sub_sft_e2bc_<domain>_dev.trec` (13/13 official-checked, de-tied).
See `results/PHASE9_PLAN.md`, `docs/RESEARCH_LOG.md` §5i/§5l."""

out = lines[:start] + new_block.split("\n") + lines[end + 1:]
io.open(path, "w", encoding="utf-8", newline="").write("\n".join(out))
print("patched lines", start + 1, "-", end + 1)

import io, json

readme = r"D:\RETECO\github_repo\README.md"
rows = json.load(open(r"D:\RETECO\results\weight_sweep_e2_dev.json", encoding="utf-8"))["rows"]

lines = io.open(readme, encoding="utf-8", newline="").read().split("\n")
start = next(i for i, l in enumerate(lines)
             if l.startswith("**The SFT arm inherited a weight tuned on the zero-shot arm.**"))
end = next(i for i, l in enumerate(lines)
           if l.startswith("**Disclosure: the SFT validation split is not recoverable.**"))
assert start < end

tbl = ["| w (CE) | zero-shot | SFT (2 epochs) | delta | 95% domain-clustered CI | domains up |",
       "|---:|---:|---:|---:|---|---:|"]
for r in rows:
    mark = "**" if abs(r["w_ce"] - 1.0) < 1e-9 else ""
    tbl.append(f"| {mark}{r['w_ce']:g}{mark} | {r['zs']:.4f} | {mark}{r['sft']:.4f}{mark} | "
               f"{mark}{r['delta']:+.4f}{mark} | "
               f"{mark}[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{mark} | {r['domains_up']}/13 |")

new_block = f"""**The SFT arm inherited a weight tuned on the zero-shot arm.** One global scalar
was fitted once on the 1,211 train queries with the zero-shot CE (1:1), frozen,
and never re-fitted; the SFT arms are therefore the *under-tuned* side, so their
measured gains are conservative rather than inflated. To show that the claim does
not hinge on that choice, `scripts/weight_sweep.py` re-fuses the stored
per-query scores at nine weights on dev, for each arm separately (diagnostic
only — nothing is selected on dev). For the frozen 2-epoch arm:

{chr(10).join(tbl)}

Every weight is positive, and at 0.7/1.5/1.8 all 13 domains improve, so the
result is not a lucky weight. The 1-epoch arm's sweep is kept as the ablation:

| w (CE) | zero-shot | SFT (1 epoch) | delta | 95% domain-clustered CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3211 | +0.0004 | [−0.0031,+0.0042] | 5/13 |
| 0.7 | 0.3294 | 0.3287 | −0.0007 | [−0.0081,+0.0053] | 7/13 |
| 0.85 | 0.3261 | 0.3308 | +0.0047 | [+0.0001,+0.0098] | 8/13 |
| **1 (frozen)** | 0.3249 | **0.3336** | **+0.0087** | **[+0.0038,+0.0142]** | 10/13 |
| 1.15 | 0.3271 | 0.3319 | +0.0048 | [+0.0008,+0.0089] | 9/13 |
| 1.3 | 0.3278 | 0.3334 | +0.0056 | [+0.0004,+0.0114] | 10/13 |
| 1.5 | 0.3253 | 0.3298 | +0.0045 | [−0.0018,+0.0108] | 8/13 |
| 1.8 | 0.3192 | 0.3341 | +0.0149 | [+0.0055,+0.0293] | 12/13 |
| 2.0 | 0.3193 | 0.3274 | +0.0080 | [+0.0041,+0.0122] | 11/13 |

There the surface is flat to ~±0.008 with no monotone trend. We report the
frozen-weight number in both cases, never a dev-best one.
(`results/WEIGHT_SWEEP_E2.md`, `results/weight_sweep_e2_dev.json`,
`results/WEIGHT_SWEEP.md`.)"""

out = lines[:start] + new_block.split("\n") + [""] + lines[end:]
io.open(readme, "w", encoding="utf-8", newline="").write("\n".join(out))
print("patched sweep section, lines", start + 1, "-", end)

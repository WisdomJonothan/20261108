import io, re, sys
LOCAL='expand_actions3_local.py'
s=io.open(LOCAL,encoding='utf-8').read()

# 1. --split / --qids-file
if '"--split"' not in s:
    s=s.replace('    ap.add_argument("--limit-queries", type=int, default=0)',
                '    ap.add_argument("--limit-queries", type=int, default=0)\n'
                '    ap.add_argument("--split", default="dev")\n'
                '    ap.add_argument("--qids-file", default="")',1)
# 2. pass split/qids into build
if 'def build(root, model, ce_tag, start_tag' in s:
    s=s.replace('def build(root, model, ce_tag, start_tag', 'def build(root, model, ce_tag, start_tag, split="dev", qids=None',1)
s=s.replace('    rows, _ = B.build_rows(root, args.candidates, DOMAINS)', '    rows, _ = B.build_rows(root, args.candidates, DOMAINS)',1)
io.open('patched_probe.py','w',encoding='utf-8',newline='\n').write(s)
print('scratch written', len(s))

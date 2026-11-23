import json,glob,os
for cell in ['sel','req','both','mis']:
    for f in ['f0','f1','f2']:
        p='results/replacement_v2/req/%s_%s.json'%(cell,f)
        if not os.path.exists(p): continue
        d=json.load(open(p))['summary']
        print(cell,f,'n',d['n_queries'],'base',round(d['macro_baseline'],4),'cell',round(d['macro_cell'],4),'d',round(d['macro_delta'],4),'swaps',d['swaps'],'h',d['helpful'],'-',d['harmful'])

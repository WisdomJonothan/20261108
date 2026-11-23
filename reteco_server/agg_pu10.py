import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd = ('cd ~/RETECO && .venv/bin/python scripts/aggregate_scores.py --pattern "results/pu10/score_*.json" '
       '--out results/aggregate_pu10_dev.json 2>&1 | tail -3 && '
       '.venv/bin/python -c "'
       'import json\n'
       'fz=json.load(open(\\"results/aggregate_frozenrepro_dev.json\\"))\n'
       'pu=json.load(open(\\"results/aggregate_pu10_dev.json\\"))\n'
       'F={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in fz[\\"domains\\"]}\n'
       'P={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in pu[\\"domains\\"]}\n'
       'print(\\"frozen %.5f  pu10 %.5f  delta %+.5f\\" % (fz[\\"domain_macro\\"], pu[\\"domain_macro\\"], pu[\\"domain_macro\\"]-fz[\\"domain_macro\\"]))\n'
       'for k in sorted(F, key=lambda x: P[x]-F[x]): print(\\"  %-12s %.4f -> %.4f  %+.4f\\" % (k,F[k],P[k],P[k]-F[k]))"')
i,o,e=c.exec_command(cmd, timeout=90)
out=o.read().decode(); err=e.read().decode()
print(out); print('ERR:', err[:300]); c.close()

import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && .venv/bin/python scripts/aggregate_scores.py --pattern "results/pu10/score_*.json" --tag pu10 --json results/aggregate_pu10_dev.json 2>&1 | tail -3 && '
     '.venv/bin/python -c "'
     'import json\n'
     'def L(p):\n'
     '    d=json.load(open(p)); return d[\\"domain_macro\\"], {x[\\"domain\\"]:x[\\"ndcg10\\"] for x in d[\\"domains\\"]}\n'
     'fm,F=L(\\"results/aggregate_frozenrepro_dev.json\\"); nm,N=L(\\"results/aggregate_nohard_dev.json\\"); pm,P=L(\\"results/aggregate_pu10_dev.json\\")\n'
     'print(\\"frozen %.4f | nohard %.4f (%+.4f) | pu10 %.4f (%+.4f)\\" % (fm,nm,nm-fm,pm,pm-fm))\n'
     'for k in sorted(F,key=lambda x:P[x]-F[x]): print(\\"  %-11s fr %.4f nh %.4f pu %.4f %+.4f\\" % (k,F[k],N[k],P[k],P[k]-F[k]))"')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()); print(e.read().decode()[:150]); c.close()

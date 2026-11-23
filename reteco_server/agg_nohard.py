import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && ls results/nohard/score_*.json | wc -l && '
     '.venv/bin/python scripts/aggregate_scores.py --pattern "results/nohard/score_*.json" --tag nohard --json results/aggregate_nohard_dev.json 2>&1 | tail -3 && '
     '.venv/bin/python -c "'
     'import json\n'
     'def L(p):\n'
     '    d=json.load(open(p)); return {x[\\"domain\\"]:round(x[\\"ndcg10\\"],4) for x in d[\\"domains\\"]}\n'
     'n=L(\\"results/aggregate_nohard_dev.json\\"); u=L(\\"results/aggregate_pu10_dev.json\\"); f=L(\\"results/aggregate_frozenrepro_dev.json\\")\n'
     'for k in sorted(n): print(\\"%-12s fr %.4f nh %.4f pu %.4f\\" % (k,f[k],n[k],u[k]))"')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()); print(e.read().decode()[:200]); c.close()

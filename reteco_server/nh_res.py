import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && .venv/bin/python scripts/aggregate_scores.py --pattern "results/nohard/score_*.json" --tag nohard --json results/aggregate_nohard_dev.json 2>&1 | tail -4 && '
     '.venv/bin/python -c "'
     'import json\n'
     'f=json.load(open(\\"results/aggregate_frozenrepro_dev.json\\")); n=json.load(open(\\"results/aggregate_nohard_dev.json\\"))\n'
     'F={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in f[\\"domains\\"]}; N={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in n[\\"domains\\"]}\n'
     'print(\\"frozen %.4f nohard %.4f delta %+.4f\\" % (f[\\"domain_macro\\"], n[\\"domain_macro\\"], n[\\"domain_macro\\"]-f[\\"domain_macro\\"]))\n'
     'for k in sorted(F,key=lambda x:N[x]-F[x]): print(\\"  %-11s %.4f -> %.4f %+.4f\\" % (k,F[k],N[k],N[k]-F[k]))"')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()); print(e.read().decode()[:200]); c.close()

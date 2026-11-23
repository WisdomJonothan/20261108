import paramiko, json
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO && .venv/bin/python -c "'
 'import json,glob\n'
 'fz=json.load(open(\\"results/aggregate_frozenrepro_dev.json\\"))\n'
 'pu=json.load(open(\\"results/aggregate_pu10_dev.json\\"))\n'
 'F={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in fz[\\"domains\\"]}\n'
 'P={d[\\"domain\\"]:d[\\"ndcg10\\"] for d in pu[\\"domains\\"]}\n'
 'import statistics\n'
 'print(\\"frozen %.5f pu10 %.5f delta %+.5f\\" % (fz[\\"domain_macro\\"], pu[\\"domain_macro\\"], pu[\\"domain_macro\\"]-fz[\\"domain_macro\\"]))\n'
 'for k in sorted(F, key=lambda k: P[k]-F[k]): print(\\"  %-12s %.4f -> %.4f  %+.4f\\" % (k,F[k],P[k],P[k]-F[k]))"', timeout=60)
print(o.read().decode()); print(e.read().decode()[:300]); c.close()

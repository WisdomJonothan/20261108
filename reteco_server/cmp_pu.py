import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && ls results/aggregate_nohard_dev.json 2>/dev/null; '
     '.venv/bin/python -c "'
     'import json,os\n'
     'def L(p):\n'
     '    d=json.load(open(p)); return {x[\\"domain\\"]:round(x[\\"ndcg10\\"],4) for x in d[\\"domains\\"]}\n'
     'p={\\"frozen\\":L(\\"results/aggregate_frozenrepro_dev.json\\"),\\"nohard\\":L(\\"results/aggregate_nohard_dev.json\\") if os.path.exists(\\"results/aggregate_nohard_dev.json\\") else {},\\"pu10\\":L(\\"results/aggregate_pu10_dev.json\\")}\n'
     'ks=sorted(p[\\"pu10\\"])\n'
     'for k in ks: print(\\"%-12s fr %.4f nh %.4f pu %.4f\\" % (k,p[\\"frozen\\"][k],p[\\"nohard\\"].get(k,0),p[\\"pu10\\"][k]))"')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()); print(e.read().decode()[:200]); c.close()

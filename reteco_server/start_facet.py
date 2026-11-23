import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && nohup setsid .venv/bin/python -u scripts/facet_recall.py '
     '--domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace '
     '--split dev --topk 100 --out results/analysis/facet_recall_dev.json > logs/facet_recall.log 2>&1 < /dev/null & '
     'sleep 5; echo LAUNCHED; tail -3 logs/facet_recall.log')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()); print(e.read().decode()[:300]); c.close()

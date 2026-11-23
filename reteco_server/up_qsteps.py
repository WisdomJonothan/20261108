import os, paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); n=0
for d in ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law","monero","politics","quant","travel","workplace"]:
    p=f'D:/RETECO/data/track1_tempo/{d}/qrels_steps_dev.txt'
    if os.path.exists(p):
        sf.put(p, f'/homeC/wujiaheng/RETECO/data/track1_tempo/{d}/qrels_steps_dev.txt'); n+=1
print('uploaded', n)
i,o,e=c.exec_command('cd ~/RETECO && nohup setsid .venv/bin/python -u scripts/facet_recall.py --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace --split dev --topk 100 --out results/analysis/facet_recall_dev.json > logs/facet_recall.log 2>&1 < /dev/null & sleep 4; echo RESTARTED')
print(o.read().decode())
c.close()

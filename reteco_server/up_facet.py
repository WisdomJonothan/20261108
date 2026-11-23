import os, paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
sf.put('facet_recall.py','/homeC/wujiaheng/RETECO/scripts/facet_recall.py')
n=0
for d in ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law","monero","politics","quant","travel","workplace"]:
    p=f'D:/RETECO/data/track1_tempo/{d}/steps_dev.jsonl'
    if os.path.exists(p):
        sf.put(p, f'/homeC/wujiaheng/RETECO/data/track1_tempo/{d}/steps_dev.jsonl'); n+=1
print('uploaded steps files', n)
c.close()

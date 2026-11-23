import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && head -8 data/track1_tempo/history/qrels_steps_train.txt; echo ===; wc -l data/track1_tempo/*/qrels_steps_train.txt | tail -3; echo ===; '
     'ls scripts | grep -i step; echo ===; grep -rn "qrels_steps" scripts/*.py | head -10')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()[:2000]); c.close()

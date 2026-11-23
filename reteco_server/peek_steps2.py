import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && head -2 data/track1_tempo/history/steps_train.jsonl; echo ===; wc -l data/track1_tempo/history/steps_train.jsonl data/track1_tempo/history/examples_train.jsonl; echo ===; '
     'for d in history economics genealogy hsm politics travel bitcoin monero cardano law quant workplace iota; do echo -n "$d "; wc -l < data/track1_tempo/$d/documents.jsonl; done')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()[:1800]); c.close()

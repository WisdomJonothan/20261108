import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && head -c 1200 data/track1_tempo/history/examples_train.jsonl; echo; echo ===; '
     '.venv/bin/python -c "'
     'import json,collections\n'
     'n=0; s=0; keys=collections.Counter(); steplen=[]\n'
     'for d in [\\"bitcoin\\",\\"history\\",\\"quant\\"]:\n'
     '    for line in open(\\"data/track1_tempo/%s/examples_train.jsonl\\"%d, encoding=\\"utf-8\\"):\n'
     '        o=json.loads(line); n+=1; keys.update(o.keys())\n'
     '        st=o.get(\\"steps\\") or []\n'
     '        s+=len(st)\n'
     '        for x in st[:1]: steplen.append(len(str(x)))\n'
     'print(\\"queries\\",n,\\"steps\\",s,\\"keys\\",dict(keys))\n'
     'print(\\"first step len\\", steplen[:5])"')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()[:2500]); print('ERR',e.read().decode()[:300]); c.close()

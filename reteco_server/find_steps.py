import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && ls data/ | head -20; echo ===; find data -maxdepth 2 -name "*step*" -o -maxdepth 2 -name "*train*" | head -20; echo ===; '
     'grep -rl "\\"steps\\"" data --include=*.json* 2>/dev/null | head -5; echo ===; '
     'ls -la data/*.json* 2>/dev/null | head')
i,o,e=c.exec_command(cmd,timeout=90); print(o.read().decode()[:2000]); c.close()

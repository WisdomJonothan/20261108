import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('zero_gain.py','/homeC/wujiaheng/RETECO/scripts/zero_gain.py')
i,o,e=c.exec_command('cd ~/RETECO && .venv/bin/python -u scripts/zero_gain.py 2>&1 | tail -15',timeout=180)
print(o.read().decode()); print(e.read().decode()[:200]); c.close()

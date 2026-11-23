import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('pool_union.py','/homeC/wujiaheng/RETECO/scripts/pool_union.py')
i,o,e=c.exec_command('cd ~/RETECO && .venv/bin/python -u scripts/pool_union.py 2>&1 | tail -20',timeout=300)
print(o.read().decode()); print(e.read().decode()[:300]); c.close()

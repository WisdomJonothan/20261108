import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO; date; ps aux | grep -c "[f]acet_pool"; tail -4 logs/facet_pool.log 2>/dev/null', timeout=30)
print(o.read().decode()); c.close()

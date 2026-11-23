import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO; date; ls -l logs/facet_recall.log 2>/dev/null; tail -6 logs/facet_recall.log 2>/dev/null; ls results/analysis/ | tail -5', timeout=30)
print(o.read().decode()); c.close()

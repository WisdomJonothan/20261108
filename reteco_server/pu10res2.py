import paramiko, json
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO && ls results/*.json | head -20 && echo --- && tail -6 logs/pu_chain.log', timeout=40)
print(o.read().decode()); c.close()

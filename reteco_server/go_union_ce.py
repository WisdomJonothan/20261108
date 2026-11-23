import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('run_union_ce2.sh','/homeC/wujiaheng/RETECO/run_union_ce2.sh')
i,o,e=c.exec_command('cd ~/RETECO && sed -i "s/\r$//" run_union_ce2.sh && bash -n run_union_ce2.sh && bash run_union_ce2.sh 2>&1 | tail -25',timeout=900)
print(o.read().decode()); print('ERR',e.read().decode()[:300]); c.close()

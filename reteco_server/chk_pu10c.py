import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cat ~/RETECO/pu_train_chain.sh; echo ===; sed -n "1,35p" ~/RETECO/ndcg_dev_chain.sh; echo ===; head -3 ~/RETECO/logs/pu10_ce.log 2>/dev/null; tail -6 ~/RETECO/logs/pu10_ce.log 2>/dev/null')
i,o,e=c.exec_command(cmd,timeout=40); print(o.read().decode()); c.close()

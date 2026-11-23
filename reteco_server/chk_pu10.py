import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('ls -d /tmp/reteco/ce_ft/*/ 2>/dev/null; echo ---; ls /tmp/reteco/ce_ft/pu10 2>/dev/null; echo ---; '
     'sed -n "1,60p" ~/RETECO/pu_chain.sh 2>/dev/null | head -40; echo ---; '
     'grep -n "model" ~/RETECO/ndcg_dev_chain.sh | head -10')
i,o,e=c.exec_command(cmd,timeout=40); print(o.read().decode()); c.close()

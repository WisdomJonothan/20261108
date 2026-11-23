import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && for T in nohard pu10; do rm -f runs/${T}_*_dev_ce.trec runs/fused_${T}_*_dev.trec; rm -rf results/$T; done; '
     'ls runs/ | grep -c "_dev_ce" ; echo CLEANED')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()); c.close()

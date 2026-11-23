import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('ls /tmp/reteco/ce_ft/pu10/pu10_ce/ ; echo ---; cat ~/RETECO/run_pu10.sh 2>/dev/null | head -30; echo ---; ls ~/RETECO/*.sh | head -30; echo ---; grep -rn "pu10" ~/RETECO/*.sh | head -20')
i,o,e=c.exec_command(cmd,timeout=40); print(o.read().decode()); c.close()

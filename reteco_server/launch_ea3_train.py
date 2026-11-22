import paramiko, time
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('ea3_train_chain.sh','RETECO/ea3_train_chain.sh'); sf.close()
cmd = ("cd /homeC/wujiaheng/RETECO && sed -i 's/\r$//' ea3_train_chain.sh && bash -n ea3_train_chain.sh && "
       "setsid nohup bash ea3_train_chain.sh > logs/ea3_train_chain.log 2>&1 < /dev/null & sleep 4; "
       "tail -3 logs/ea3_train_chain.log; ls -l logs/ea3_train_f0.log 2>/dev/null | head -2")
i,o,e=c.exec_command(cmd, timeout=40); time.sleep(8); print(o.read().decode()); c.close()

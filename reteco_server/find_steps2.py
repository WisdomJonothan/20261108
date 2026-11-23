import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && ls; echo ===; ls data/track1_tempo | head; echo ===; ls data/track1_tempo/history; echo ===; '
     'grep -l "steps" -r data/track1_tempo/history 2>/dev/null | head -3')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()[:1500]); c.close()

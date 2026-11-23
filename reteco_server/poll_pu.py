import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO; date; tail -2 logs/pu_chain.log; ls results/pu10/score_*.json 2>/dev/null | wc -l; ls -l results/aggregate_pu10_dev.json 2>/dev/null; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader', timeout=30)
print(o.read().decode()); c.close()

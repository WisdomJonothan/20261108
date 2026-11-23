import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd = ('cd ~/RETECO && .venv/bin/python scripts/aggregate_scores.py --pattern "results/pu10/score_*.json" '
       '--tag pu10 --json results/aggregate_pu10_dev.json 2>&1 | tail -12')
i,o,e=c.exec_command(cmd, timeout=90)
print(o.read().decode()); c.close()

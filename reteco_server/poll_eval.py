import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
i,o,e=c.exec_command('cd ~/RETECO; date; tail -3 logs/pu_eval_chain.log; for T in nohard pu10; do echo -n "$T ce:"; ls runs/${T}_*_dev_ce.trec 2>/dev/null | wc -l; echo -n " score:"; ls results/$T/score_*.json 2>/dev/null | wc -l; done', timeout=30)
print(o.read().decode()); c.close()

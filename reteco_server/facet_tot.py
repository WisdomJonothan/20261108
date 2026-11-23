import paramiko, json
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); p='D:/RETECO/github_repo/results/analysis/facet_recall_dev.json'
sf.get('/homeC/wujiaheng/RETECO/results/analysis/facet_recall_dev.json', p); c.close()
d=json.load(open(p,encoding='utf-8'))
tot=dict(q=0,qh=0,st=0,sth=0,miss=0,fix=0)
print(f'{"domain":12s} {"q":>4s} {"q_hit":>6s} {"miss":>5s} {"fixed":>6s} {"rate":>6s}')
for k,v in d.items():
    tot['q']+=v['queries']; tot['qh']+=v['query_hit']; tot['st']+=v['steps']
    tot['sth']+=v['step_hit']; tot['miss']+=v['query_miss']; tot['fix']+=v['query_miss_fixed_by_step']
    r=v['query_miss_fixed_by_step']/v['query_miss'] if v['query_miss'] else 0
    print(f'{k:12s} {v["queries"]:4d} {v["query_hit"]:6d} {v["query_miss"]:5d} {v["query_miss_fixed_by_step"]:6d} {r:6.2f}')
print(f'{"TOTAL":12s} {tot["q"]:4d} {tot["qh"]:6d} {tot["miss"]:5d} {tot["fix"]:6d} {tot["fix"]/tot["miss"]:6.2f}')
print('steps', tot['st'], 'step_hit', tot['sth'])

import paramiko, json
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); p='D:/RETECO/github_repo/results/analysis/hardneg_probe.json'
sf.get('/homeC/wujiaheng/RETECO/results/analysis/hardneg_probe.json', p); c.close()
d=json.load(open(p))
print('n', d['n'], 'thresholds', d['thresholds'])
import statistics
ps=[r['p_yes'] for r in d['records']]
print('mean p_yes %.3f median %.3f' % (statistics.mean(ps), statistics.median(ps)))
from collections import defaultdict
byrank=defaultdict(list)
for r in d['records']: byrank[min(10,r['dense_rank']//2*2)].append(r['p_yes'])
for k in sorted(byrank): print('dense rank', k, 'n', len(byrank[k]), 'mean p %.3f' % statistics.mean(byrank[k]))

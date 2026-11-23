import paramiko, os
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
files={
 'results/aggregate_nohard_dev.json':'results/aggregate_nohard_dev.json',
 'results/aggregate_pu10_dev.json':'results/aggregate_pu10_dev.json',
 'results/analysis/facet_recall_dev.json':'results/analysis/facet_recall_dev.json',
 'results/analysis/facet_pool_dev.json':'results/analysis/facet_pool_dev.json',
 'results/analysis/pool_union_dev.json':'results/analysis/pool_union_dev.json',
 'results/analysis/zero_facet_gain.json':'results/analysis/zero_facet_gain.json',
 'results/analysis/zero_facet_ce.json':'results/analysis/zero_facet_ce.json',
}
base='D:/RETECO/github_repo/'
for remote, local in files.items():
    p=os.path.join(base, local); os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        sf.get('/homeC/wujiaheng/RETECO/'+remote, p)
        print('ok', local, os.path.getsize(p))
    except Exception as e:
        print('FAIL', remote, e)
c.close()

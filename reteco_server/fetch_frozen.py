import paramiko, os
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
for n in ['aggregate_frozenrepro_dev.json']:
    p=f'D:/RETECO/github_repo/results/{n}'
    sf.get('/homeC/wujiaheng/RETECO/results/'+n, p); print('ok', n, os.path.getsize(p))
c.close()

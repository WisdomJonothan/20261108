import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.get('/homeC/wujiaheng/RETECO/results/analysis/neg_audit.json','D:/RETECO/github_repo/results/analysis/neg_audit.json')
print('ok'); c.close()

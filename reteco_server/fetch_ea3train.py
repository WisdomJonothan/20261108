import paramiko, os
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
dst='D:/RETECO/github_repo/results/replacement_v2/'
for f in ['f0','f1','f2']:
    name='expanded3_fixed_train_%s.json'%f
    sf.get('/homeC/wujiaheng/RETECO/results/replacement_v2/'+name, dst+name)
    print(name, os.path.getsize(dst+name))
c.close()

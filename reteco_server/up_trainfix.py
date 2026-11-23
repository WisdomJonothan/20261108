import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
sf.put('D:/RETECO/github_repo/scripts/train_ce.py','/homeC/wujiaheng/RETECO/scripts/train_ce.py')
i,o,e=c.exec_command('cd ~/RETECO && cp scripts/train_ce.py /tmp/train_ce_fix.py && grep -n "rank.unsqueeze(1) < k) | (rank" scripts/train_ce.py && .venv/bin/python -c "import ast;ast.parse(open(\'scripts/train_ce.py\').read());print(\'AST_OK\')"',timeout=60)
print(o.read().decode()); print(e.read().decode()[:300]); c.close()

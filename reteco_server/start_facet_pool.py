import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('facet_pool.py','/homeC/wujiaheng/RETECO/scripts/facet_pool.py')
cmd=('cd ~/RETECO && nohup setsid .venv/bin/python -u scripts/facet_pool.py --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace --split dev --topk 50 --out-tag facetstep > logs/facet_pool.log 2>&1 < /dev/null & sleep 4; echo LAUNCHED')
i,o,e=c.exec_command(cmd,timeout=40); print(o.read().decode()); c.close()

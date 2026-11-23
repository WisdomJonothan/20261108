import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp()
p='/homeC/wujiaheng/RETECO/ndcg_dev_chain.sh'
s=sf.open(p).read().decode()
old='''if [ "$CASE" = "frozen" ]; then
  CKPT=results/ce_ft/v2m3_pool8_e2/best
else
  CKPT=/tmp/reteco/ce_ft/ndcg/ndcg_ce/best
fi'''
new='''if [ "$CASE" = "frozen" ]; then
  CKPT=results/ce_ft/v2m3_pool8_e2/best
else
  CKPT=/tmp/reteco/ce_ft/${TAG}/${TAG}_ce/best
  if [ ! -f "$CKPT/config.json" ]; then echo "MISSING CKPT $CKPT"; exit 1; fi
fi'''
assert old in s, 'pattern not found'
s=s.replace(old,new)
f=sf.open(p,'w'); f.write(s); f.close()
print('patched', p)
i,o,e=c.exec_command('bash -n ~/RETECO/ndcg_dev_chain.sh && echo SYNTAX_OK', timeout=20)
print(o.read().decode(), e.read().decode()[:200])
c.close()

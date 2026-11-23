import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
sf=c.open_sftp(); sf.put('mk_union_run.py','/homeC/wujiaheng/RETECO/scripts/mk_union_run.py')
cmd=('cd ~/RETECO && .venv/bin/python -u scripts/mk_union_run.py && '
     'wc -l runs/union_zero_dev.trec && '
     'CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_chunk_rerank_qvar.py --domain history --split dev '
     '--run runs/union_zero_dev.trec --k 200 --q-mode title --q-cap-words 0 --pair-max-len 510 --q-tokens 256 '
     '--win 250 --stride 192 --pool max --model results/ce_ft/v2m3_pool8_e2/best '
     '--out /tmp/reteco/union_zero_ce.trec --meta /tmp/reteco/union_zero.meta.json --batch 16 2>&1 | tail -8')
i,o,e=c.exec_command(cmd,timeout=600); print(o.read().decode()); print('ERR',e.read().decode()[:300]); c.close()

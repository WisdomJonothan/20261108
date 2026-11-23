import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.13.71.195',username='wujiaheng',password='wjh123456',timeout=20)
cmd=('cd ~/RETECO && cat > pu_eval_chain.sh <<"EOS"\n'
     'cd "$HOME/RETECO" || exit 1\n'
     'for T in nohard pu10; do\n'
     '  echo "=== eval $T $(date +%H:%M) ==="\n'
     '  bash ndcg_dev_chain.sh 0 $T 2>&1 | tail -2\n'
     'done\n'
     'echo PU_EVAL_DONE\n'
     'EOS\n'
     'nohup setsid bash pu_eval_chain.sh > logs/pu_eval_chain.log 2>&1 < /dev/null &\n'
     'sleep 3; echo LAUNCHED; tail -2 logs/pu_eval_chain.log; ls -l /tmp/reteco/ce_ft/nohard/nohard_ce/best/config.json /tmp/reteco/ce_ft/pu10/pu10_ce/best/config.json')
i,o,e=c.exec_command(cmd,timeout=60); print(o.read().decode()); print(e.read().decode()[:200]); c.close()

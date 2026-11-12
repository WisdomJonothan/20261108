$bash = "C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"
$cmd = 'cd /d/RETECO && CKPT=results/ce_ft/v2m3_pool8_e2s29/best TAG=sft_e2s29 PYTHONUNBUFFERED=1 bash scripts/phase9_dev_rerank.sh > logs/detached_rerank29.out 2>&1'
$p = Start-Process -FilePath $bash -ArgumentList ('-lc "{0}"' -f $cmd) -WindowStyle Hidden -PassThru
Write-Output ("launched pid " + $p.Id)

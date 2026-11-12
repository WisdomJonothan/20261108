$bash = "C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"
$cmd = "cd /d/RETECO && PYTHONUNBUFFERED=1 PYTHONUTF8=1 /d/anaconda/python.exe scripts/train_ce.py --model models/bge-reranker-v2-m3 --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace --groups-prefix results/ce_train_groups --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 --epochs 2 --lr 1.5e-5 --seed 29 --tag v2m3_pool8_e2s29 --outdir results/ce_ft > logs/detached_e2s29b.out 2>&1"
$p = Start-Process -FilePath $bash -ArgumentList ('-lc "{0}"' -f $cmd) -WindowStyle Hidden -PassThru
$p.Id | Out-File -Encoding ascii "D:\RETECO\logs\detached_e2s29b.pid"
Write-Output $p.Id

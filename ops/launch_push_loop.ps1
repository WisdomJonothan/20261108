$bash = "C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"
$cmd = 'cd /d/RETECO && mkdir -p logs && bash ops/push_loop.sh > /dev/null 2>&1'
$p = Start-Process -FilePath $bash -ArgumentList ('-lc "{0}"' -f $cmd) -WindowStyle Hidden -PassThru
Write-Output ("push loop pid " + $p.Id)

$bash = "C:\Users\Lenovo\.dsh\profiles\desktop\node_modules\@dsh-ops\bash-win32-x64\usr\bin\bash.exe"
$cmd = 'cd /d/RETECO && bash ops/queue_after_s29.sh'
$p = Start-Process -FilePath $bash -ArgumentList ('-lc "{0}"' -f $cmd) -WindowStyle Hidden -PassThru
Write-Output ("launched pid " + $p.Id)

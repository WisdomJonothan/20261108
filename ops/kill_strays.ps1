$needles = @('phase11', 'phase9_dev_rerank', 'v2m3_pool8_e1b', 'phase9_after_train', 'phase11_oof_finish', 'phase11_finish_eval')
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($needles | Where-Object { $_.CommandLine -like ("*" + $_ + "*") }) }
foreach ($p in $procs) {
  Write-Output ("kill {0} {1}" -f $p.ProcessId, $p.Name)
  try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch { Write-Output ("  failed: " + $_.Exception.Message) }
}

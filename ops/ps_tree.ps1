$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('bash.exe','powershell.exe','pwsh.exe','python.exe','node.exe') }
foreach ($p in $procs) {
  $cl = $p.CommandLine
  if ($cl -and $cl.Length -gt 220) { $cl = $cl.Substring(0,220) }
  Write-Output ("{0} | ppid={1} | {2} | {3} | {4}" -f $p.ProcessId, $p.ParentProcessId, $p.CreationDate, $p.Name, $cl)
}

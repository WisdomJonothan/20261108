param([int]$TargetPid)
$p = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $TargetPid)
if (-not $p) { Write-Output "no such process"; exit 0 }
Write-Output ("PID     = " + $p.ProcessId)
Write-Output ("PPID    = " + $p.ParentProcessId)
Write-Output ("START   = " + $p.CreationDate)
Write-Output ("CMD     = " + $p.CommandLine)
$q = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $p.ParentProcessId)
if ($q) { Write-Output ("PARENT  = " + $q.Name + " :: " + $q.CommandLine) }

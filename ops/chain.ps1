param([int]$StartPid)
$cur = $StartPid
for ($i = 0; $i -lt 8; $i++) {
  $p = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $cur)
  if (-not $p) { Write-Output "  (gone at $cur)"; break }
  $cl = $p.CommandLine
  if ($cl -and $cl.Length -gt 160) { $cl = $cl.Substring(0, 160) }
  Write-Output ("{0} ppid={1} {2} :: {3}" -f $p.ProcessId, $p.ParentProcessId, $p.CreationDate, $cl)
  $cur = $p.ParentProcessId
  if ($cur -eq 0) { break }
}

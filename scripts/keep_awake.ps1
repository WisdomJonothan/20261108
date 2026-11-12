# Hold the machine out of idle sleep for a bounded window.
#
# Why: on 2026-11-12 02:18 the laptop went to sleep for 7 s in the middle of a
# dev re-rank.  On resume the CUDA context was dead, but the process did not
# error out: it sat at 86 % GPU utilisation for 11 h with zero progress (see
# logs/final_lx_possum_genealogy.log).  Any GPU queue longer than the idle
# timeout needs this guard.
#
# SetThreadExecutionState is per-process, so closing this PowerShell lets the
# normal power policy back in; nothing is written to the power scheme.
param([double]$Hours = 8)

Add-Type -Namespace Reteco -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

# NB: the flags must be built as [uint32].  Literal 0x80000000 parses as a
# negative Int32 in PowerShell and the P/Invoke call then throws
# (CannotConvertArgument), which silently kills this script on its first pass;
# `powercfg /requests` is the only way to see whether the guard is holding.
$ES_CONTINUOUS      = [uint32]2147483648   # 0x80000000
$ES_KEEP_ON         = [uint32]2147483649   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$deadline = (Get-Date).AddHours($Hours)
Write-Host ("keep_awake: holding system awake until {0:HH:mm}" -f $deadline)
while ((Get-Date) -lt $deadline) {
    [void][Reteco.Power]::SetThreadExecutionState($ES_KEEP_ON)
    Start-Sleep -Seconds 60
}
[void][Reteco.Power]::SetThreadExecutionState($ES_CONTINUOUS)
Write-Host "keep_awake: released"

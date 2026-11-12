# Prints BUSY when another python process is using the GPU for one of the
# project's model-loaded jobs, FREE otherwise.
#
# Why not nvidia-smi: on this laptop WDDM keeps ~30 desktop context pids in the
# compute-apps list and a baseline of 1-3 GB allocated even when idle, so
# memory alone cannot tell a real job from the desktop.
$pat = 'run_chunk_rerank|run_dense_generic|train_ce|encode_docs|score_ce|run_pairwise'
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match $pat) }
if ($busy) {
  $busy | ForEach-Object { Write-Output ("BUSY " + $_.ProcessId + " " + $_.CommandLine) }
  exit 0
}
Write-Output "FREE"

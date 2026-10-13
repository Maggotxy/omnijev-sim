param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$SshHost = 'bxj',
    [ValidateRange(1024,65535)][int]$Port = 8790
)
$ErrorActionPreference = 'Stop'
$jevUrl = "http://127.0.0.1:$Port"
try {
    $jevHealth = Invoke-RestMethod -Uri "$jevUrl/state" -TimeoutSec 2
    if ($jevHealth.status) { Write-Output "Ready: $jevUrl"; return }
} catch { }
$jevLogDir = Join-Path $PSScriptRoot '..\out\tunnel'
New-Item -ItemType Directory -Force -Path $jevLogDir | Out-Null
$jevLogId = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$jevSsh = (Get-Command ssh -ErrorAction Stop).Source
$jevProcess = Start-Process -FilePath $jevSsh -WindowStyle Hidden -PassThru -ArgumentList @(
    '-N', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-L',
    "${Port}:127.0.0.1:8790", $SshHost
) -RedirectStandardOutput (Join-Path $jevLogDir "$jevLogId.out.log") `
  -RedirectStandardError (Join-Path $jevLogDir "$jevLogId.err.log")
for ($jevAttempt = 0; $jevAttempt -lt 10; $jevAttempt++) {
    Start-Sleep -Milliseconds 500
    if ($jevProcess.HasExited) { throw "SSH forwarding failed. See $jevLogDir\$jevLogId.err.log" }
    try {
        $jevHealth = Invoke-RestMethod -Uri "$jevUrl/state" -TimeoutSec 2
        if ($jevHealth.status) { Write-Output "Ready: $jevUrl (SSH PID $($jevProcess.Id))"; return }
    } catch { }
}
Stop-Process -Id $jevProcess.Id -ErrorAction SilentlyContinue
throw 'The loopback simulation service did not respond. Start jev_reflex.live on the server, then retry.'

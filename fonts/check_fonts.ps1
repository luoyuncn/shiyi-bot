# List installed JetBrains font families
[System.Reflection.Assembly]::LoadWithPartialName('System.Drawing') | Out-Null
$fonts = (New-Object System.Drawing.Text.InstalledFontCollection).Families
$jb = $fonts | Where-Object { $_.Name -match 'JetBrains|Nerd' }
if ($jb) {
    Write-Host "`nFound matching fonts:" -ForegroundColor Green
    foreach ($f in $jb) {
        Write-Host "  -> $($f.Name)" -ForegroundColor Cyan
    }
} else {
    Write-Host "`nNo JetBrains or Nerd fonts found!" -ForegroundColor Red
}
Write-Host ""

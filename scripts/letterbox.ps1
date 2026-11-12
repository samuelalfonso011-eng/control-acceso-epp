# Convierte fotos a entrada del modelo (letterbox gris 114, RGB crudo) para scripts/diag-fotos.mjs
# Uso: powershell -File scripts/letterbox.ps1 -Src "C:\ruta\fotos" -Dst "D:\tmp\diag" -Size 416
param([string]$Src, [string]$Dst, [int]$Size = 416)
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $Dst | Out-Null
$meta = @()
Get-ChildItem $Src -File | Where-Object { $_.Extension -match '^\.(jpe?g|png|bmp)$' } | ForEach-Object {
  $img = [System.Drawing.Image]::FromFile($_.FullName)
  $w = $img.Width; $h = $img.Height
  $scale = [Math]::Min($Size / $w, $Size / $h)
  $nw = [int][Math]::Round($w * $scale); $nh = [int][Math]::Round($h * $scale)
  $px = [int][Math]::Floor(($Size - $nw) / 2); $py = [int][Math]::Floor(($Size - $nh) / 2)
  $bmp = New-Object System.Drawing.Bitmap $Size, $Size, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::FromArgb(114, 114, 114))
  $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear
  $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
  $g.DrawImage($img, $px, $py, $nw, $nh)
  $g.Dispose(); $img.Dispose()
  $rect = New-Object System.Drawing.Rectangle 0, 0, $Size, $Size
  $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, $bmp.PixelFormat)
  $stride = $data.Stride
  $buf = New-Object byte[] ($stride * $Size)
  [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $buf, 0, $buf.Length)
  $bmp.UnlockBits($data); $bmp.Dispose()
  $rgb = New-Object byte[] ($Size * $Size * 3)
  for ($y = 0; $y -lt $Size; $y++) {
    $row = $y * $stride; $o = $y * $Size * 3
    for ($x = 0; $x -lt $Size; $x++) {
      $i = $row + $x * 3; $j = $o + $x * 3
      $rgb[$j] = $buf[$i + 2]; $rgb[$j + 1] = $buf[$i + 1]; $rgb[$j + 2] = $buf[$i]
    }
  }
  [System.IO.File]::WriteAllBytes((Join-Path $Dst ($_.BaseName + '.rgb')), $rgb)
  $meta += [pscustomobject]@{ name = $_.Name; file = $_.BaseName + '.rgb'; srcW = $w; srcH = $h; scale = $scale; padX = $px; padY = $py }
}
$meta | ConvertTo-Json | Set-Content (Join-Path $Dst 'meta.json') -Encoding UTF8
"$($meta.Count) fotos -> $Dst"

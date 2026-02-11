# Download Qdrant Windows binary
$version = "v1.13.2"
$url = "https://github.com/qdrant/qdrant/releases/download/$version/qdrant-x86_64-pc-windows-msvc.zip"
$output = "$PSScriptRoot\qdrant.zip"

Write-Host "Downloading Qdrant $version..."
Invoke-WebRequest -Uri $url -OutFile $output

Write-Host "Extracting..."
Expand-Archive -Path $output -DestinationPath $PSScriptRoot -Force

Write-Host "Cleaning up..."
Remove-Item $output

Write-Host "Done! Qdrant binary: $PSScriptRoot\qdrant.exe"

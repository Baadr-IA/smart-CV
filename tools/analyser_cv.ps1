param([string]$filename)
$url = "http://localhost:8000/analyze-local?filename=$([uri]::EscapeDataString($filename))&generate_word=true"
$response = Invoke-RestMethod -Uri $url -Method Post
$response | ConvertTo-Json -Depth 10

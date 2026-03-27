param([string]$query)
$url = "http://localhost:8000/search?query=$([uri]::EscapeDataString($query))"
$response = Invoke-RestMethod -Uri $url -Method Get
$response | ConvertTo-Json -Depth 10

# Personal Knowledge Agent - start backend
# Sets PYTHONUTF8=1 to fix GBK/UTF-8 encoding errors on Chinese Windows.
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
.venv\Scripts\langgraph.exe dev --port 3001

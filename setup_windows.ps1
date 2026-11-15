$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Find-Python311 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = & py -3.11 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        'C:\Program Files\Python311\python.exe'
    )
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        $version = & $command.Source -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>$null
        if ($LASTEXITCODE -eq 0 -and $version -eq '3.11') { return $command.Source }
    }
    return $null
}

try {
    $python = Find-Python311
    if (-not $python) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Python 3.11 nao foi encontrado e winget nao esta disponivel. Instale Python 3.11 em https://www.python.org/downloads/release/python-3119/ e execute este script novamente.'
        }
        Write-Host 'Instalando Python 3.11 pelo winget...'
        & winget install --id Python.Python.3.11 --exact --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw 'A instalacao do Python falhou. Verifique a mensagem do winget.' }
        $python = Find-Python311
        if (-not $python) { throw 'Python foi instalado, mas esta janela ainda nao o encontra. Feche-a, abra outra e execute setup_windows.bat novamente.' }
    }
    Write-Host "Usando Python: $python"
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel criar o ambiente virtual.' }
    }
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $venvPython -m pip --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'pip nao encontrado no ambiente virtual. Tentando instala-lo com ensurepip...'
        & $venvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel instalar pip no ambiente virtual.' }
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Nao foi possivel atualizar pip. Tentando instalar as bibliotecas com a versao disponivel.'
        & $venvPython -m pip --version
        if ($LASTEXITCODE -ne 0) { throw 'pip nao esta funcionando no ambiente virtual.' }
    }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as bibliotecas.' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'As bibliotecas instaladas apresentam conflito.' }
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
        Write-Host 'Arquivo .env criado.'
    }
    # Em um clone do Git, faz o Git ignorar alteracoes locais no .env,
    # para que a chave colada nele nunca seja enviada ao GitHub por engano.
    if ((Test-Path -LiteralPath '.git') -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $trackedEnv = & git ls-files -- .env
        if ($LASTEXITCODE -eq 0 -and $trackedEnv) {
            & git update-index --skip-worktree .env
            if ($LASTEXITCODE -eq 0) { Write-Host 'Protecao do .env ativada: o Git vai ignorar a chave que voce colar nele.' }
        }
    }
    Write-Host 'Abra o arquivo .env e cole sua chave em GOOGLE_API_KEY= (veja a secao 2 do README).'
    Write-Host 'Instalacao concluida. Depois de configurar .env, use abrir_versao_atual.bat.'
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}

<#
.SYNOPSIS
Build a patched FidelityFX Vulkan runtime DLL and vendor it into the repo.

.DESCRIPTION
The stock FidelityFX SDK v1.1.4 ships its FSR3 upscaler GLSL shader with the
`rw_luma_history` storage image declared `rgba8`, while the C++ creates the
resource as R16G16B16A16_FLOAT. On Vulkan this trips a validation-layer
format-mismatch warning every FSR dispatch (upstream issue #161, still open).

This script applies the one-line fix (`rgba8` -> `rgba16f`), rebuilds the
ffx-api Vulkan DLL from SDK source (which recompiles the shader permutations),
and copies the result to concinnity-client/third_party/ffx/, where build.rs
prefers it over the stock SDK copy.

It is idempotent: re-running re-applies the patch only if needed and rebuilds.

.PARAMETER SdkRoot
FidelityFX SDK v1.1.4 source root. Defaults to $env:FIDELITYFX_SDK_ROOT, then
C:\FidelityFX-SDK-v1.1.4.

.PARAMETER CloneIfMissing
If the SDK source is absent, git-clone the v1.1.4 tag into SdkRoot first.

.PARAMETER Generator
CMake generator. Defaults to "Visual Studio 18 2026".

.EXAMPLE
pwsh scripts/setup_ffx_vk_dll.ps1
.EXAMPLE
pwsh scripts/setup_ffx_vk_dll.ps1 -SdkRoot D:\ffx -CloneIfMissing
#>
[CmdletBinding()]
param(
    [string]$SdkRoot = $(if ($env:FIDELITYFX_SDK_ROOT) { $env:FIDELITYFX_SDK_ROOT } else { 'C:\FidelityFX-SDK-v1.1.4' }),
    [string]$Configuration = 'Release',
    [string]$Generator = 'Visual Studio 18 2026',
    [switch]$CloneIfMissing
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RepoRoot 'concinnity-client\third_party\ffx'
$VendorDll = Join-Path $VendorDir 'amd_fidelityfx_vk.dll'
$ShaderHeader = Join-Path $SdkRoot 'sdk\include\FidelityFX\gpu\fsr3upscaler\ffx_fsr3upscaler_callbacks_glsl.h'
$FfxApiDir = Join-Path $SdkRoot 'ffx-api'
$BuildDir = Join-Path $FfxApiDir 'build-vk-concinnity'
$BuiltDll = Join-Path $FfxApiDir 'bin\amd_fidelityfx_vk.dll'   # Release has no name postfix
$FfxRepoTag = 'v1.1.4'
$FfxRepoUrl = 'https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git'

function Require-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$name' not found on PATH."
    }
}

Write-Host "FidelityFX VK DLL setup" -ForegroundColor Cyan
Write-Host "  SDK root : $SdkRoot"
Write-Host "  Vendor   : $VendorDll"

# 1. Obtain the SDK source.
if (-not (Test-Path $ShaderHeader)) {
    if ($CloneIfMissing) {
        Require-Tool git
        Write-Host "SDK not found; cloning $FfxRepoTag into $SdkRoot ..." -ForegroundColor Yellow
        git clone --depth 1 --branch $FfxRepoTag $FfxRepoUrl $SdkRoot
        if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
    } else {
        throw "FidelityFX SDK source not found at $SdkRoot (missing $ShaderHeader). " +
              "Install the SDK, set FIDELITYFX_SDK_ROOT, or pass -CloneIfMissing."
    }
}

Require-Tool cmake
if (-not $env:VULKAN_SDK) {
    Write-Warning "VULKAN_SDK is not set; shader compilation (glslc) may fail."
}

# 2. Apply the rgba8 -> rgba16f fix to the luma_history UAV declaration.
$content = Get-Content -Raw -LiteralPath $ShaderHeader
$pattern = '(binding = FSR3UPSCALER_BIND_UAV_LUMA_HISTORY, )rgba8(\) uniform image2D\s+rw_luma_history)'
if ($content -match 'rgba16f\) uniform image2D\s+rw_luma_history') {
    Write-Host "Shader already patched (rgba16f)." -ForegroundColor Green
} elseif ($content -match $pattern) {
    $backup = "$ShaderHeader.orig"
    if (-not (Test-Path $backup)) { Copy-Item -LiteralPath $ShaderHeader $backup }
    $patched = $content -replace $pattern, '${1}rgba16f${2}'
    Set-Content -LiteralPath $ShaderHeader -Value $patched -NoNewline
    Write-Host "Patched rw_luma_history: rgba8 -> rgba16f (backup at *.orig)." -ForegroundColor Green
} else {
    throw "Could not locate the rw_luma_history rgba8 declaration in $ShaderHeader. " +
          "The SDK layout may differ from v1.1.4."
}

# 3. Configure + build the ffx-api Vulkan DLL (pulls in ../sdk, recompiles shaders).
Write-Host "Configuring ffx-api (VK_X64) ..." -ForegroundColor Cyan
cmake -S $FfxApiDir -B $BuildDir -G $Generator -A x64 -DFFX_API_BACKEND=VK_X64
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

Write-Host "Building ffx-api ($Configuration) ..." -ForegroundColor Cyan
cmake --build $BuildDir --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

if (-not (Test-Path $BuiltDll)) {
    throw "Build succeeded but $BuiltDll was not produced. Check the build output."
}

# 4. Vendor the DLL into the repo.
if (-not (Test-Path $VendorDir)) { New-Item -ItemType Directory -Path $VendorDir | Out-Null }
Copy-Item -LiteralPath $BuiltDll -Destination $VendorDll -Force

$size = (Get-Item $VendorDll).Length
Write-Host ""
Write-Host "Done. Vendored patched DLL ($([math]::Round($size/1MB,2)) MB):" -ForegroundColor Green
Write-Host "  $VendorDll"
Write-Host "build.rs will prefer it over the stock SDK copy on the next --features vulkan build."

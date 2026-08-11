param(
    [Parameter(Mandatory = $true)]
    [string[]]$Reference,

    [Parameter(Mandatory = $true)]
    [string[]]$Output,

    [string[]]$SourceName,
    [string]$Python = "python",
    [string]$Model = "wav2vec2",
    [int]$Layer = 2,
    [double]$Alpha = 1.0,
    [int]$Seed = 42,
    [int]$MaxGpus = 0,
    [ValidateSet("error", "trim")]
    [string]$LengthPolicy = "error",
    [string]$ResultsDir = "mapss_results",
    [switch]$NoCI,
    [switch]$Plot,
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

if ($Reference.Count -lt 2) {
    throw "MAPSS requires at least two reference/output sources."
}
if ($Reference.Count -ne $Output.Count) {
    throw "Found $($Reference.Count) references and $($Output.Count) outputs."
}
if ($SourceName -and $SourceName.Count -ne $Reference.Count) {
    throw "SourceName must contain one unique name per source."
}
if ($Plot -and $NoCI) {
    throw "Plot cannot be combined with NoCI because the paper-style figure includes confidence quantities."
}

foreach ($Path in ($Reference + $Output)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Audio file does not exist: $Path"
    }
}

$MapssArguments = @(
    "-m", "mapss", "--reference"
) + $Reference + @(
    "--output"
) + $Output + @(
    "--model", $Model,
    "--layer", "$Layer",
    "--alpha", "$Alpha",
    "--seed", "$Seed",
    "--max-gpus", "$MaxGpus",
    "--length-policy", $LengthPolicy,
    "--results-dir", $ResultsDir
)

foreach ($Name in $SourceName) {
    $MapssArguments += @("--source-name", $Name)
}
if ($NoCI) {
    $MapssArguments += "--no-ci"
}
if ($Plot) {
    $MapssArguments += "--plot"
}
if ($VerboseOutput) {
    $MapssArguments += "--verbose"
}

& $Python @MapssArguments
if ($LASTEXITCODE -ne 0) {
    throw "MAPSS failed with exit code $LASTEXITCODE."
}

Write-Host "MAPSS completed successfully. Results: $ResultsDir"

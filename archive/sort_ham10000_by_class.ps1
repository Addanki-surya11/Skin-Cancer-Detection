param(
    [string]$BaseDir = ".",
    [string]$OutputDir = "sorted_by_dx",
    [switch]$Force,
    [switch]$ReconcileOnly
)

$ErrorActionPreference = "Stop"

$basePath = Resolve-Path $BaseDir
$metadataPath = Join-Path $basePath "HAM10000_metadata.csv"
$outPath = Join-Path $basePath $OutputDir

if (-not (Test-Path $metadataPath)) {
    throw "Metadata file not found: $metadataPath"
}

if (-not (Test-Path $outPath)) {
    New-Item -ItemType Directory -Path $outPath | Out-Null
}

# HAM10000 diagnosis code descriptions.
$labelMap = @{
    "akiec" = "akiec"
    "bcc"   = "bcc"
    "bkl"   = "bkl"
    "df"    = "df"
    "mel"   = "mel"
    "nv"    = "nv"
    "vasc"  = "vasc"
}

$rows = Import-Csv $metadataPath
$metaByImageId = @{}
foreach ($row in $rows) {
    $metaByImageId[$row.image_id] = $row
}

$sourceCandidates = @(
    (Join-Path $basePath "HAM10000_images_part_1"),
    (Join-Path $basePath "HAM10000_images_part_2")
)
$sourceDirs = $sourceCandidates | Where-Object { Test-Path $_ }

function Ensure-ClassDirs {
    param(
        [string]$Root,
        [hashtable]$Map
    )

    foreach ($dx in $Map.Keys) {
        $classDir = Join-Path $Root $Map[$dx]
        if (-not (Test-Path $classDir)) {
            New-Item -ItemType Directory -Path $classDir | Out-Null
        }
    }
}

function Run-Reconcile {
    param(
        [string]$Root,
        [hashtable]$Map,
        [hashtable]$Meta,
        [switch]$ForceMove
    )

    Ensure-ClassDirs -Root $Root -Map $Map

    $checked = 0
    $moved = 0
    $missingInMetadata = 0

    Get-ChildItem $Root -Recurse -File -Filter "*.jpg" | ForEach-Object {
        $checked++
        $file = $_
        $imageId = $file.BaseName

        if (-not $Meta.ContainsKey($imageId)) {
            $missingInMetadata++
            return
        }

        $trueDx = $Meta[$imageId].dx
        if (-not $Map.ContainsKey($trueDx)) {
            return
        }

        $targetDir = Join-Path $Root $Map[$trueDx]
        $targetPath = Join-Path $targetDir $file.Name
        $currentDirName = Split-Path $file.DirectoryName -Leaf

        if ($currentDirName -ne $Map[$trueDx]) {
            if ($ForceMove -or -not (Test-Path $targetPath)) {
                Move-Item -Path $file.FullName -Destination $targetPath -Force
                $moved++
            }
        }
    }

    $counts = Get-ChildItem $Root -Directory | Sort-Object Name | ForEach-Object {
        [PSCustomObject]@{
            Class = $_.Name
            Count = (Get-ChildItem $_.FullName -File -Filter "*.jpg" | Measure-Object).Count
        }
    }

    $total = ($counts | Measure-Object Count -Sum).Sum

    Write-Output "Mode: reconcile-current"
    Write-Output "Checked present files: $checked"
    Write-Output "Moved to correct class: $moved"
    Write-Output "Present files not found in metadata: $missingInMetadata"
    Write-Output "Current sorted counts:"
    $counts | Format-Table -AutoSize | Out-String | Write-Output
    Write-Output "Current sorted total: $total"
}

$total = 0
$missing = 0
$copiedByClass = @{}

Ensure-ClassDirs -Root $outPath -Map $labelMap

foreach ($dx in $labelMap.Keys) {
    $copiedByClass[$dx] = 0
}

if ($ReconcileOnly -or $sourceDirs.Count -eq 0) {
    Run-Reconcile -Root $outPath -Map $labelMap -Meta $metaByImageId -ForceMove:$Force
    if ($sourceDirs.Count -eq 0 -and -not $ReconcileOnly) {
        Write-Output "Source image folders not found. Reconcile mode was used for present files only."
    }
    return
}

$imagePathLookup = @{}
foreach ($srcDir in $sourceDirs) {
    Get-ChildItem $srcDir -File -Filter "*.jpg" | ForEach-Object {
        $imagePathLookup[$_.BaseName] = $_.FullName
    }
}

foreach ($row in $rows) {
    $imageId = $row.image_id
    $dx = $row.dx

    if (-not $labelMap.ContainsKey($dx)) {
        continue
    }

    if (-not $imagePathLookup.ContainsKey($imageId)) {
        $missing++
        continue
    }

    $fileName = "$imageId.jpg"
    $src = $imagePathLookup[$imageId]

    $dst = Join-Path (Join-Path $outPath $labelMap[$dx]) $fileName

    if ($Force) {
        Copy-Item -Path $src -Destination $dst -Force
        $copiedByClass[$dx]++
    } else {
        if (-not (Test-Path $dst)) {
            Copy-Item -Path $src -Destination $dst
            $copiedByClass[$dx]++
        }
    }

    $total++
}

Run-Reconcile -Root $outPath -Map $labelMap -Meta $metaByImageId -ForceMove:$Force

Write-Output "Mode: sort-from-sources"
Write-Output "Processed rows: $($rows.Count)"
Write-Output "Resolved image paths: $total"
Write-Output "Missing images: $missing"
Write-Output "Copied per class:"
$copiedByClass.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Output "  $($_.Name): $($_.Value)"
}

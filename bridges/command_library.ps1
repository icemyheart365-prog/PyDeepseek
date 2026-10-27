# Native Windows command backend.
#
# Emits the same JSON envelope as python_backend.py so the caller cannot tell
# which backend ran. Safety rules mirror the Python backend:
#   * paths are resolved against library/ and must stay inside it
#   * symlinks/reparse points that escape the root are rejected
#   * protected file names are refused
#   * no command string is ever concatenated; every check is done in-process
#
# Reads one JSON request from stdin: {"args": ["ls", ".", "name", "false"]}
# and writes one JSON envelope to stdout as UTF-8.

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LibraryRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptRoot 'library'))

function Write-Envelope {
    param(
        [string]$Status,
        [int]$ExitCode,
        [byte[]]$Stdout,
        [byte[]]$Stderr
    )
    $toBase64 = {
        param([byte[]]$Bytes)
        if ($null -eq $Bytes -or $Bytes.Length -eq 0) { return '' }
        return [Convert]::ToBase64String($Bytes)
    }
    $payload = [ordered]@{
        status       = $Status
        exit_code    = $ExitCode
        stdout_bytes = if ($null -eq $Stdout) { 0 } else { $Stdout.Length }
        stderr_bytes = if ($null -eq $Stderr) { 0 } else { $Stderr.Length }
        stdout_b64   = & $toBase64 $Stdout
        stderr_b64   = & $toBase64 $Stderr
        truncated    = $false
        backend      = 'powershell'
    }
    $json = ConvertTo-Json -InputObject $payload -Compress
    $out = [Console]::OpenStandardOutput()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json + "`n")
    $out.Write($bytes, 0, $bytes.Length)
    $out.Flush()
}

function Fail {
    param([string]$Message, [int]$Code = 2)
    Write-Envelope -Status 'error' -ExitCode $Code -Stdout $null -Stderr ([System.Text.Encoding]::UTF8.GetBytes($Message))
    exit 0
}

$ProtectedNames = @('config.json', '.env', 'credentials', 'secrets', 'global.json')

function Resolve-LibraryPath {
    param([string]$Relative)
    if ([string]::IsNullOrEmpty($Relative)) { $Relative = '.' }
    if ($Relative -match '[\\]') { throw 'Invalid or inaccessible path' }
    if ([System.IO.Path]::IsPathRooted($Relative)) { throw 'Invalid or inaccessible path' }
    if (($Relative -split '/') -contains '..') { throw 'Invalid or inaccessible path' }

    $candidate = [System.IO.Path]::GetFullPath((Join-Path $LibraryRoot $Relative))
    $rootWithSep = $LibraryRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if ($candidate -ne $LibraryRoot -and -not $candidate.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Invalid or inaccessible path'
    }

    # Resolve every component before the containment check so junctions and
    # symlinks cannot escape library/. ResolveLinkTarget only inspects the leaf,
    # so walk each ancestor and follow it as well.
    $checkPath = $candidate
    $visited = @{}
    while ($true) {
        $exists = $false
        try { $exists = Test-Path -LiteralPath $checkPath } catch { $exists = $false }
        if (-not $exists) { break }

        $item = $null
        try { $item = Get-Item -LiteralPath $checkPath -Force -ErrorAction Stop } catch {
            throw 'Invalid or inaccessible path'
        }
        if ($item.LinkType -or ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            $resolved = $null
            try { $resolved = $item.ResolveLinkTarget($true) } catch { $resolved = $null }
            if ($null -eq $resolved) { throw 'Invalid or inaccessible path' }
            $physical = [System.IO.Path]::GetFullPath($resolved.FullName)
            if ($visited.ContainsKey($physical)) { throw 'Invalid or inaccessible path' }
            $visited[$physical] = $true
            # Any reparse point that leaves the library root is refused outright.
            $checkPath = $physical
            if ($checkPath -ne $LibraryRoot -and
                -not $checkPath.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Invalid or inaccessible path'
            }
        }
        $parent = Split-Path -Parent $checkPath
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $checkPath) { break }
        $checkPath = $parent
    }

    $name = Split-Path -Leaf $candidate
    if ($ProtectedNames -contains $name) { throw 'Access to this path is blocked' }
    return $candidate
}

function Get-BytesFromText {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return [byte[]]::new(0) }
    return [System.Text.Encoding]::UTF8.GetBytes($Text)
}

function Get-TextFromFile {
    param([string]$Path)
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Get-HumanSize {
    param([long]$Size)
    $units = @('B', 'K', 'M', 'G', 'T')
    $value = [double]$Size
    $index = 0
    while ($value -ge 1024 -and $index -lt ($units.Length - 1)) {
        $value /= 1024
        $index++
    }
    if ($index -eq 0) { return "$([int]$value)$($units[$index])" }
    return ('{0:F1}{1}' -f $value, $units[$index])
}

$raw = [Console]::In.ReadToEnd()
try {
    $request = $raw | ConvertFrom-Json
    $args = @($request.args)
} catch {
    Fail 'Invalid JSON request'
}

if ($args.Count -lt 1) { Fail 'Missing command name' }
$cmd = [string]$args[0]
$rest = @()
if ($args.Count -gt 1) { $rest = @($args[1..($args.Count - 1)]) }

if (-not (Test-Path -LiteralPath $LibraryRoot -PathType Container)) {
    Fail 'library/ directory does not exist'
}

try {
    switch ($cmd) {
        'ls' {
            $rel = if ($rest.Count -ge 1) { [string]$rest[0] } else { '.' }
            $fmt = if ($rest.Count -ge 2) { [string]$rest[1] } else { 'name' }
            $all = if ($rest.Count -ge 3) { [string]$rest[2] } else { 'false' }
            if ($fmt -notin @('name', 'long', 'human', 'long_human')) { throw 'Invalid ls format or include_all flag' }
            if ($all -notin @('true', 'false')) { throw 'Invalid ls format or include_all flag' }

            $target = Resolve-LibraryPath $rel
            if (-not (Test-Path -LiteralPath $target -PathType Container)) { throw 'A directory is required' }

            $entries = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction Stop |
                Where-Object { $all -eq 'true' -or -not $_.Name.StartsWith('.') } |
                Sort-Object { $_.Name.ToLowerInvariant() })

            $lines = foreach ($entry in $entries) {
                $isDir = $entry.PSIsContainer
                $size = 0
                if (-not $isDir) { $size = $entry.Length }
                $kind = if ($isDir) { 'd' } else { '-' }
                $sizeText = if ($fmt -in @('human', 'long_human')) { Get-HumanSize $size } else { [string]$size }

                if ($fmt -in @('long', 'long_human')) {
                    '{0} {1,8} {2}' -f $kind, $sizeText, $entry.Name
                } elseif ($fmt -eq 'human') {
                    '{0,8} {1}' -f $sizeText, $entry.Name
                } else {
                    $entry.Name
                }
            }
            $text = if ($null -eq $lines) { '' } else { (($lines -join "`n") + "`n") }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'stat' {
            if ($rest.Count -ne 1) { throw 'Usage: stat <relative-path>' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target)) { throw 'Path does not exist' }
            $item = Get-Item -LiteralPath $target -Force
            $type = if ($item.PSIsContainer) { 'directory' } else { 'file' }
            $size = if ($item.PSIsContainer) { 0 } else { $item.Length }
            $text = "type=$type`nsize_bytes=$size`n"
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'sha256' {
            if ($rest.Count -ne 1) { throw 'Usage: sha256 <relative-path>' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            $text = "$hash  $target`n"
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'cat' {
            if ($rest.Count -ne 1) { throw 'Usage: cat <relative-path>' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            $bytes = [System.IO.File]::ReadAllBytes($target)
            if ($bytes.Length -gt 131072) { Fail 'File exceeds cat size limit' 3 }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout $bytes -Stderr $null
            exit 0
        }
        'wc' {
            $rel = if ($rest.Count -ge 1) { [string]$rest[0] } else { '' }
            $mode = if ($rest.Count -ge 2) { [string]$rest[1] } else { 'all' }
            $target = Resolve-LibraryPath $rel
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            if ($mode -notin @('bytes', 'lines', 'words', 'all')) { throw 'Invalid wc mode' }

            $linesText = [System.IO.File]::ReadAllLines($target, [System.Text.Encoding]::UTF8)
            $byteCount = ([System.IO.FileInfo]$target).Length
            $lineCount = $linesText.Length
            $wordCount = 0
            foreach ($line in $linesText) {
                $wordCount += @($line -split '\s+' | Where-Object { $_ -ne '' }).Count
            }
            $text = switch ($mode) {
                'bytes' { "$byteCount`n" }
                'lines' { "$lineCount`n" }
                'words' { "$wordCount`n" }
                default { "$lineCount $wordCount $byteCount`n" }
            }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'head' {
            if ($rest.Count -lt 1 -or $rest.Count -gt 2) { throw 'Usage: head <relative-path> [lines]' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            $n = if ($rest.Count -ge 2) { [string]$rest[1] } else { '20' }
            if ($n -notmatch '^\d+$') { throw 'Line count must be an integer' }
            $count = [int]$n
            if ($count -lt 1 -or $count -gt 1000) { throw 'Line count outside allowed range' }

            $all = [System.IO.File]::ReadAllLines($target, [System.Text.Encoding]::UTF8)
            $take = [Math]::Min($count, $all.Length)
            $text = if ($take -gt 0) { (($all[0..($take - 1)] -join "`n") + "`n") } else { '' }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'tail' {
            if ($rest.Count -lt 1 -or $rest.Count -gt 2) { throw 'Usage: tail <relative-path> [lines]' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            $n = if ($rest.Count -ge 2) { [string]$rest[1] } else { '20' }
            if ($n -notmatch '^\d+$') { throw 'Line count must be an integer' }
            $count = [int]$n
            if ($count -lt 1 -or $count -gt 1000) { throw 'Line count outside allowed range' }

            $all = [System.IO.File]::ReadAllLines($target, [System.Text.Encoding]::UTF8)
            $take = [Math]::Min($count, $all.Length)
            $text = if ($take -gt 0) {
                (($all[($all.Length - $take)..($all.Length - 1)] -join "`n") + "`n")
            } else { '' }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'grep' {
            if ($rest.Count -lt 2 -or $rest.Count -gt 4) { throw 'Usage: grep <relative-path> <pattern> [ignore_case] [max_matches]' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'A regular file is required' }
            $pattern = [string]$rest[1]
            $ignore = if ($rest.Count -ge 3) { [string]$rest[2] } else { 'false' }
            $limitText = if ($rest.Count -ge 4) { [string]$rest[3] } else { '100' }
            if ($pattern.Length -gt 256) { throw 'Pattern too long' }
            if ($ignore -notin @('true', 'false')) { throw 'ignore_case must be true or false' }
            if ($limitText -notmatch '^\d+$') { throw 'max_matches must be an integer' }
            $limit = [int]$limitText
            if ($limit -lt 1 -or $limit -gt 1000) { throw 'max_matches out of range' }

            $comparison = if ($ignore -eq 'true') {
                [System.StringComparison]::OrdinalIgnoreCase
            } else {
                [System.StringComparison]::Ordinal
            }
            $matched = [System.Collections.Generic.List[string]]::new()
            foreach ($line in [System.IO.File]::ReadAllLines($target, [System.Text.Encoding]::UTF8)) {
                if ($line.IndexOf($pattern, $comparison) -ge 0) {
                    $matched.Add($line)
                    if ($matched.Count -ge $limit) { break }
                }
            }
            $text = if ($matched.Count -gt 0) { (($matched -join "`n") + "`n") } else { '' }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'search_files' {
            if ($rest.Count -lt 2 -or $rest.Count -gt 3) { throw 'Usage: search_files <relative-directory> <filename-pattern> [recursive]' }
            $target = Resolve-LibraryPath ([string]$rest[0])
            if (-not (Test-Path -LiteralPath $target -PathType Container)) { throw 'A directory is required' }
            $pattern = [string]$rest[1]
            $recursive = if ($rest.Count -ge 3) { [string]$rest[2] } else { 'false' }
            if ($pattern.Length -gt 128) { throw 'Pattern too long' }
            if ($recursive -notin @('true', 'false')) { throw 'recursive must be true or false' }

            $found = if ($recursive -eq 'true') {
                @(Get-ChildItem -LiteralPath $target -Recurse -File -Force -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like $pattern })
            } else {
                @(Get-ChildItem -LiteralPath $target -File -Force -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like $pattern })
            }
            $paths = @($found | ForEach-Object { $_.FullName } | Sort-Object)
            $text = if ($paths.Count -gt 0) { (($paths -join "`n") + "`n") } else { '' }
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText $text) -Stderr $null
            exit 0
        }
        'pwd' {
            Write-Envelope -Status 'ok' -ExitCode 0 -Stdout (Get-BytesFromText "library/`n") -Stderr $null
            exit 0
        }
        default {
            Fail 'Unknown or unsupported command'
        }
    }
} catch {
    Fail $_.Exception.Message
}

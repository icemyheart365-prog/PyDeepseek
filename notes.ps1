<#
.SYNOPSIS
    Sync HANDOFF.md between machines through the notes branch.

.DESCRIPTION
    HANDOFF.md is ignored on main but tracked on the notes branch. Switching
    branches makes git delete tracked files the target branch lacks, so a naive
    switch would remove your notes. This script always restores the working copy
    afterwards.

    Run it from the repository root on any branch.

.PARAMETER Action
    pull    Fetch the latest notes from origin and write it to disk.
    push    Commit the current HANDOFF.md on notes and push it.
    status  Show whether the local copy matches origin/notes (default).

.EXAMPLE
    .\notes.ps1 pull      # at school, before editing
    .\notes.ps1 push      # at home, after editing
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('pull', 'push', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'

$NotesBranch = 'notes'
$NotesFile = 'HANDOFF.md'

function Get-CurrentBranch {
    return (& git rev-parse --abbrev-ref HEAD).Trim()
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    # Only stdout is captured: git writes progress and CRLF notices to stderr, and
    # funnelling stderr into the pipeline turns warnings into terminating errors.
    $out = & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)`n$out"
    }
    return $out
}

function Save-NotesFromBranch {
    param([string]$Ref)
    # Copy the blob byte-for-byte: PowerShell pipelines and Start-Process redirects
    # both decode with the console code page, which corrupts non-ASCII text.
    $target = Join-Path (Get-Location) $NotesFile
    $spec = "${Ref}:${NotesFile}"
    $cmd = 'git show "{0}" > "{1}"' -f $spec, $target
    & $env:ComSpec /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "cannot read $NotesFile from $spec" }
}

function Enter-Notes {
    $original = Get-CurrentBranch
    if ($original -eq $NotesBranch) { return $original }
    $dirty = & git status --porcelain --untracked-files=no
    if ($dirty) {
        throw "Working tree has tracked changes. Commit or stash them first:`n$dirty"
    }
    Invoke-Git switch $NotesBranch | Out-Null
    return $original
}

function Exit-Notes {
    param([string]$Original, [string]$Ref)
    if ($Original -and $Original -ne $NotesBranch) {
        Invoke-Git switch $Original | Out-Null
    }
    # Switching may delete the file (it is tracked on notes, ignored elsewhere).
    Save-NotesFromBranch $Ref
}

switch ($Action) {
    'pull' {
        # Never clobber uncommitted edits: a pull silently discarding local work is
        # exactly how notes get lost.
        if (Test-Path $NotesFile) {
            $localHash = (& git hash-object $NotesFile).Trim()
            & git fetch origin $NotesBranch --quiet 2>$null | Out-Null
            $remoteHash = (& git rev-parse --verify --quiet "origin/${NotesBranch}:${NotesFile}").Trim()
            if ($remoteHash -and $localHash -ne $remoteHash) {
                Write-Host "Refusing to pull: $NotesFile has local changes." -ForegroundColor Yellow
                Write-Host "  Run '.\notes.ps1 push' first, or move the file aside to discard it." -ForegroundColor Yellow
                return
            }
        }

        $original = Enter-Notes
        try {
            Invoke-Git fetch origin $NotesBranch | Out-Null
            $behind = (& git rev-list --count "HEAD..origin/${NotesBranch}").Trim()
            Invoke-Git merge --ff-only "origin/${NotesBranch}" | Out-Null
            Write-Host "Pulled origin/${NotesBranch} ($behind new commit(s))." -ForegroundColor Green
        }
        finally {
            Exit-Notes $original $NotesBranch
        }
        Write-Host "Updated $NotesFile ($((Get-Item $NotesFile).Length) bytes)." -ForegroundColor Green
    }

    'push' {
        if (-not (Test-Path $NotesFile)) { throw "$NotesFile not found." }

        # Switching to notes overwrites the working copy with the branch version,
        # so stash the edited file first and restore it once on the branch.
        $backup = Join-Path ([System.IO.Path]::GetTempPath()) "handoff-$PID.bak"
        Copy-Item $NotesFile $backup -Force

        $original = Enter-Notes
        try {
            # Refuse to build on a stale branch: if origin moved ahead, the push
            # would fail as non-fast-forward and the local edit could be lost.
            Invoke-Git fetch origin $NotesBranch | Out-Null
            $behind = (& git rev-list --count "HEAD..origin/${NotesBranch}").Trim()
            if ($behind -ne '0') {
                Write-Host "Refusing to push: origin/$NotesBranch has $behind commit(s) you do not have." -ForegroundColor Yellow
                Write-Host "  Run '.\notes.ps1 pull' first, merge by hand, then push." -ForegroundColor Yellow
                return
            }

            Copy-Item $backup $NotesFile -Force
            Invoke-Git add -f $NotesFile | Out-Null
            $staged = (& git diff --cached --name-only)
            if (-not $staged) {
                Write-Host "No changes to $NotesFile; nothing to push." -ForegroundColor Yellow
                return
            }
            Invoke-Git commit -m "notes: update $NotesFile" | Out-Null
            Invoke-Git push origin $NotesBranch | Out-Null
            Write-Host "Pushed $NotesFile to origin/$NotesBranch." -ForegroundColor Green
        }
        finally {
            Remove-Item $backup -Force -ErrorAction SilentlyContinue
            Exit-Notes $original $NotesBranch
        }
    }

    'status' {
        Write-Host "Branch : $(Get-CurrentBranch)"
        if (Test-Path $NotesFile) {
            Write-Host "File   : $NotesFile ($((Get-Item $NotesFile).Length) bytes)"
        }
        else {
            Write-Host "File   : missing (run: .\notes.ps1 pull)" -ForegroundColor Yellow
        }

        & git fetch origin $NotesBranch --quiet 2>$null | Out-Null
        $remoteHash = (& git rev-parse --verify --quiet "origin/${NotesBranch}:${NotesFile}").Trim()
        if (-not $remoteHash) {
            Write-Host "Remote : no copy on origin/$NotesBranch yet" -ForegroundColor Yellow
            return
        }
        if (-not (Test-Path $NotesFile)) { return }

        # Compare blob hashes: immune to CRLF and console-encoding differences.
        $localHash = (& git hash-object $NotesFile).Trim()
        Write-Host ""
        if ($localHash -eq $remoteHash) {
            Write-Host "Remote vs local: identical" -ForegroundColor Green
        }
        else {
            Write-Host "Remote vs local: differs (run pull or push)" -ForegroundColor Yellow
        }
    }
}

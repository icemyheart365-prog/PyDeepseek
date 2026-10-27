#!/bin/sh
# Deepseek command library v4
# Raw-output backend: Python captures stdout/stderr and constructs the JSON envelope.
# Do not pass arbitrary shell text. Arguments must be validated by the Python dispatcher.

set -u

SCRIPT_DIR=$(dirname "$0")
case "$SCRIPT_DIR" in
  /*) ;;
  *) SCRIPT_DIR="$(pwd)/$SCRIPT_DIR" ;;
esac
LIBRARY_ROOT="$SCRIPT_DIR/library"

error() {
  echo "ERROR: $2" >&2
  exit "$1"
}

[ -d "$LIBRARY_ROOT" ] || error 2 "library/ directory does not exist"
[ "$#" -ge 1 ] || error 2 "Missing command name"
cmd=$1
shift

resolve_path() {
  rel=$1
  case "$rel" in
    /*) return 1 ;;
    *\\*) return 1 ;;
  esac
  case "/$rel/" in
    */../*) return 1 ;;
  esac
  [ -n "$rel" ] || rel=.
  candidate="$LIBRARY_ROOT/$rel"
  # Require the target or its parent to exist; reject symlink escape with physical pwd.
  if [ -d "$candidate" ]; then
    physical=$(cd "$candidate" 2>/dev/null && pwd -P) || return 1
    case "$physical/" in "$LIBRARY_ROOT/"*) echo "$physical"; return 0 ;; *) return 1 ;; esac
  fi
  parent=$(dirname "$candidate")
  base=$(basename "$candidate")
  physical=$(cd "$parent" 2>/dev/null && pwd -P) || return 1
  case "$physical/" in "$LIBRARY_ROOT/"*) echo "$physical/$base"; return 0 ;; *) return 1 ;; esac
}

check_protected() {
  path=$1
  base=$(basename "$path")
  case "$base" in
    config.json|.env|credentials|secrets|global.json) return 0 ;;
  esac
  return 1
}

case "$cmd" in
  ls)
    path_arg=${1:-.}; format=${2:-name}; all=${3:-false}
    resolved=$(resolve_path "$path_arg") || error 2 "Invalid or inaccessible path"
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    case "$format:$all" in
      name:false) exec ls -1 "$resolved" ;;
      name:true) exec ls -1a "$resolved" ;;
      long:false) exec ls -l "$resolved" ;;
      long:true) exec ls -la "$resolved" ;;
      human:false) exec ls -h "$resolved" ;;
      human:true) exec ls -ha "$resolved" ;;
      long_human:false) exec ls -lh "$resolved" ;;
      long_human:true) exec ls -lha "$resolved" ;;
      *) error 2 "Invalid ls format or include_all flag" ;;
    esac
    ;;
  stat)
    [ "$#" -eq 1 ] || error 2 "Usage: stat <relative-path>"
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -e "$resolved" ] || error 2 "Path does not exist"
    exec stat "$resolved"
    ;;
  wc)
    [ "$#" -ge 1 ] && [ "$#" -le 2 ] || error 2 "Usage: wc <relative-path> [bytes|lines|words|all]"
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -f "$resolved" ] || error 2 "A regular file is required"
    case "${2:-all}" in
      bytes) exec wc -c "$resolved" ;;
      lines) exec wc -l "$resolved" ;;
      words) exec wc -w "$resolved" ;;
      all) exec wc -c -l -w "$resolved" ;;
      *) error 2 "Invalid wc mode" ;;
    esac
    ;;
  sha256)
    [ "$#" -eq 1 ] || error 2 "Usage: sha256 <relative-path>"
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -f "$resolved" ] || error 2 "A regular file is required"
    # A-Shell advertises shasum; use it directly instead of probing with command -v.
    exec shasum -a 256 "$resolved"
    ;;
  head|tail|cat)
    if [ "$cmd" = cat ]; then
      [ "$#" -eq 1 ] || error 2 "Usage: cat <relative-path>"
    else
      [ "$#" -ge 1 ] && [ "$#" -le 2 ] || error 2 "Usage: $cmd <relative-path> [lines]"
    fi
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -f "$resolved" ] || error 2 "A regular file is required"
    if [ "$cmd" = cat ]; then
      size=$(wc -c < "$resolved")
      [ "$size" -le 131072 ] || error 3 "File exceeds cat size limit"
      exec cat "$resolved"
    fi
    n=${2:-20}
    case "$n" in ''|*[!0-9]*) error 2 "Line count must be an integer" ;; esac
    [ "$n" -ge 1 ] && [ "$n" -le 1000 ] || error 2 "Line count outside allowed range"
    if [ "$cmd" = head ]; then exec head -n "$n" "$resolved"; else exec tail -n "$n" "$resolved"; fi
    ;;
  grep)
    [ "$#" -ge 2 ] && [ "$#" -le 4 ] || error 2 "Usage: grep <relative-path> <pattern> [ignore_case] [max_matches]"
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    pattern=$2; ignore=${3:-false}; limit=${4:-100}
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -f "$resolved" ] || error 2 "A regular file is required"
    [ "${#pattern}" -le 256 ] || error 2 "Pattern too long"
    case "$limit" in ''|*[!0-9]*) error 2 "max_matches must be an integer" ;; esac
    [ "$limit" -ge 1 ] && [ "$limit" -le 1000 ] || error 2 "max_matches out of range"
    case "$ignore" in
      true) exec grep -i -F -m "$limit" "$pattern" "$resolved" ;;
      false) exec grep -F -m "$limit" "$pattern" "$resolved" ;;
      *) error 2 "ignore_case must be true or false" ;;
    esac
    ;;
  search_files)
    [ "$#" -ge 2 ] && [ "$#" -le 3 ] || error 2 "Usage: search_files <relative-directory> <filename-pattern> [recursive]"
    resolved=$(resolve_path "$1") || error 2 "Invalid or inaccessible path"
    pattern=$2; recursive=${3:-false}
    check_protected "$resolved" && error 2 "Access to this path is blocked"
    [ -d "$resolved" ] || error 2 "A directory is required"
    [ "${#pattern}" -le 128 ] || error 2 "Pattern too long"
    case "$recursive" in
      true) exec find "$resolved" -type f -name "$pattern" -print ;;
      false) exec find "$resolved" -maxdepth 1 -type f -name "$pattern" -print ;;
      *) error 2 "recursive must be true or false" ;;
    esac
    ;;
  *)
    error 2 "Unknown or unsupported command"
    ;;
esac

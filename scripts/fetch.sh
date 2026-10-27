#!/bin/bash
# Checks the versions of versions.conf out into sources/<name>: one shared
# repository (sources/oneDNN.git), a detached worktree per version at its
# pinned commit. Only the pinned commits are fetched (shallow).
#
# Usage: fetch.sh [--repo NAME=URL]... [name...|all]
#
#   name           versions to fetch (default: all)
#   --repo N=URL   fetch version N from URL instead (a mirror, a local clone)
#
# A worktree already at its commit is left as is; one at another commit is
# an error (remove sources/<name> to fetch it again).
#
# Examples:
#   ./fetch.sh
#   ./fetch.sh --repo rvjit=$HOME/src/oneDNN rvjit

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SOURCES_DIR=${SOURCES_DIR:-$root_dir/sources}
declare -A urls=()
names=()
while [ $# -gt 0 ]; do
    case $1 in
        --repo) urls[${2%%=*}]=${2#*=}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option $1" ;;
        all) shift ;;
        *) names+=("$1"); shift ;;
    esac
done

# Prints "<name> <url> <commit>" for each version asked for
versions() {
    local name url commit found
    for want in "${names[@]:-}"; do
        found=0
        while read -r name url commit; do
            [ -z "$want" ] || [ "$want" = "$name" ] || continue
            echo "$name ${urls[$name]:-$url} $commit"
            found=1
        done < <(grep -Ev '^[[:space:]]*(#|$)' "$root_dir/versions.conf")
        [ $found = 1 ] || die "no version '$want' in versions.conf"
    done
}

repo=$SOURCES_DIR/oneDNN.git
mkdir -p "$SOURCES_DIR"
[ -d "$repo" ] || git init -q --bare "$repo"

while read -r name url commit; do
    dir=$SOURCES_DIR/$name
    if [ -e "$dir/.git" ]; then
        have=$(git -C "$dir" rev-parse HEAD)
        [ "$have" = "$commit" ] \
                || die "$dir is at $have, not $commit (remove it to fetch again)"
        echo "=== $name: $commit, already checked out"
        continue
    fi
    phase_begin fetch "$name" "$url $commit"
    echo "=== $name: $url $commit"
    if ! git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$repo" fetch -q --depth 1 "$url" "$commit"
    fi
    git -C "$repo" worktree prune
    git -C "$repo" worktree add -q --detach "$dir" "$commit"
    phase_end 0
done < <(versions)
phase_report

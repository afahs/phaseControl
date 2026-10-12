#!/bin/sh
# Run as rrrodriguez13, after SSH key login has been tested.
set -eu
if [ "$(id -un)" != rrrodriguez13 ]; then
    echo 'Run this as rrrodriguez13.' >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git
fi
project="$HOME/projects/new-project"
if [ ! -d "$project/.git" ]; then
    if [ -e "$project" ]; then
        echo "Refusing to initialize an existing directory: $project" >&2
        exit 1
    fi
    mkdir -p "$project"
    git init "$project"
    git -C "$project" symbolic-ref HEAD refs/heads/main
fi
if [ "$(git -C "$project" symbolic-ref HEAD)" != refs/heads/main ]; then
    echo 'Expected the Pi checkout to be on main; stopping.' >&2
    exit 1
fi
git -C "$project" config receive.denyCurrentBranch updateInstead
git -C "$project" config receive.denyNonFastForwards true
git -C "$project" config receive.denyDeletes true
printf '\nReady: %s\nPush main to update the project files.\n' "$project"

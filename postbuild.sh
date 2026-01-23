#!/bin/bash
set -e

cd_prefix="./computational_methods_*"
dirs=( $cd_prefix )
if [ ${#dirs[@]} -eq 1 ] && [ -d "${dirs[0]}" ]; then
    cd "${dirs[0]}"
else
    echo "Error: expected exactly one directory matching $cd_prefix, found ${#dirs[@]}"
fi

uv venv
uv sync
source ./.venv/bin/activate
cd ..
python -m ipykernel install --user --name hwrs640

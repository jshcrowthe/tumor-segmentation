#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

pushd "$parent_path"
rm -rf data/processed

mkdir -p data/processed
touch data/processed/.gitkeep

popd
#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"
rm -rf data/processed

mkdir -p data/processed
touch data/processed/.gitkeep
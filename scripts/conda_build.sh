#!/bin/sh -e

conda build -c pytorch -c conda-forge --user "$CONDA_USER" --token "$CONDA_TOKEN" "$@"

#!/bin/sh -e

conda mambabuild -c pytorch -c conda-forge --user "$CONDA_USER" --token "$CONDA_TOKEN" "$@"

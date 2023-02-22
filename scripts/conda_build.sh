#!/bin/sh -e

conda mambabuild -c pytorch -c conda-forge --user "$CONDA_USER" --label silicon --token "$CONDA_TOKEN" "$@"

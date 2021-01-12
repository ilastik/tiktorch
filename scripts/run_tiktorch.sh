#!/bin/sh

export PREFIX=$(dirname "$(readlink -f $0)")

$PREFIX/bin/python -m tiktorch.server $@

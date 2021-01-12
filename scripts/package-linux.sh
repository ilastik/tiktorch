#/bin/sh -e
ROOT_DIR="$(dirname "$(readlink -f "$0")")"

if [ -z "$1" ]; then
    echo "No version supplied";
    exit 1;
fi
VERSION=$1

conda create --yes --name tiktorch-server-env-pkg -c ilastik-forge -c pytorch -c conda-forge tiktorch="$VERSION"
conda pack --name tiktorch-server-env-pkg --output "tiktorch-$VERSION-linux.tar" --arcroot "tiktorch-$VERSION"
tar --transform "s/^/tiktorch-$VERSION\//" -rvf "tiktorch-$VERSION-linux.tar" -C "$ROOT_DIR" "run_tiktorch.sh"
conda env remove --yes --name tiktorch-server-env-pkg
pbzip2 --best --read "tiktorch-$VERSION-linux.tar"

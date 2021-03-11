#/bin/sh -e
ROOT_DIR="$(dirname "$(readlink -f "$0")")"

if [ -z "$1" ]; then
    echo "No version supplied";
    exit 1;
fi
VERSION="$1"
FLAVOR="${2:-cpu}"
PACKAGE="tiktorch"

case $FLAVOR in
  cuda)
    PACKAGE="tiktorch"
    ;;

  cpu)
    PACKAGE="tiktorch-cpu"
    ;;

  *)
    echo -n "Unknown flavor: $FLAVOR"
    exit 1;
    ;;
esac
conda create --yes --name tiktorch-server-env-pkg -c ilastik-forge -c pytorch -c conda-forge  "$PACKAGE"="$VERSION"
conda pack --name tiktorch-server-env-pkg --output "$PACKAGE-$VERSION-linux.tar" --arcroot "tiktorch"
tar --transform "s/^/tiktorch\//" -rvf "$PACKAGE-$VERSION-linux.tar" -C "$ROOT_DIR" "run_tiktorch.sh"
conda env remove --yes --name tiktorch-server-env-pkg
pbzip2 --best --read "$PACKAGE-$VERSION-linux.tar"

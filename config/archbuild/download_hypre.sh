#!/bin/sh

# requires curl

HYPRE_VERSION=2.22.1
HYPRE_TARBALL=hypre-${HYPRE_VERSION}.tar.gz

curl -L \
  https://codeload.github.com/hypre-space/hypre/tar.gz/refs/tags/v${HYPRE_VERSION} \
  > ${HYPRE_TARBALL}

#!/bin/sh

mkdir -p lib && cd lib
ln -sf ../build/*/libmetis/libmetis.a ./lib
ln -sf ../build/*/libmetis/libmetis.a ./libmetis.a
ln -sf ../build/*/libmetis/libmetis.so ./libmetis.so

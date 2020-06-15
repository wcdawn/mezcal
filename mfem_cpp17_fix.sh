#!/bin/bash

DIR=$(dirname "$(realpath $0)")
FNAME=$DIR/mfem/general/mem_manager.hpp

sed -i 's/static inline T \*New(std::size_t)/static inline T *New(std::size_t size)/' $FNAME

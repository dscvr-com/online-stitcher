#!/bin/bash

set -e

if [ ! -d "build" ]; then
    mkdir build
fi
rm -rd dbg
mkdir dbg
cd build
cmake .. $@
make $MAKEOPTS 

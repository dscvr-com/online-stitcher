#!/bin/bash

set -e

if [ ! -d "build" ]; then
    mkdir build
fi
if [ -d "dbg" ]; then
    rm -rd dbg
fi
if [ -d "tmp" ]; then
    rm -rd tmp
fi
mkdir dbg
mkdir tmp
cd build
cmake .. $@
make $MAKEOPTS 

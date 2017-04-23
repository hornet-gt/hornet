#!/bin/bash
if [ ! -e ./cloc ]; then
    echo "./cloc does not exist."
    exit 1
fi
if [ ! -x ./cloc ]; then
    chmod +x ./cloc
fi

./cloc ../include/ ../src/
if [ $? -ne 0 ]; then exit 1; fi

./cloc --by-file ../include/
./cloc --by-file ../src/

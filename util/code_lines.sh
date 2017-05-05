#!/bin/bash
dir=`dirname $0`
ABS_PATH=`cd "$dir"; pwd`

if [ ! -e "$dir/cloc" ]; then
    echo "./cloc does not exist."
    exit 1
fi
if [ ! -x "$dir/cloc" ]; then
    chmod +x "$dir/cloc"
fi

"$dir/cloc" $dir/../include/ $dir/../src/ $dir/../test/ $dir/../externals/cuStinger/include/ $dir/../externals/cuStinger/src/
if [ $? -ne 0 ]; then exit 1; fi

cd "$ABS_PATH/../include/"
"$ABS_PATH/cloc" --by-file .

cd "$ABS_PATH/../src/"
"$ABS_PATH/cloc" --by-file .

cd "$ABS_PATH/../test/"
"$ABS_PATH/cloc" --by-file .

cd "$ABS_PATH/../externals/cuStinger/include/"
"$ABS_PATH/cloc" --by-file .

cd "$ABS_PATH/../externals/cuStinger/src/"
"$ABS_PATH/cloc" --by-file .

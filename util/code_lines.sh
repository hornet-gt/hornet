#!/bin/bash
rel_dir=`dirname $0`
dir=`cd "$rel_dir"; pwd`

if [ ! -e "$dir/cloc" ]; then
    echo "./cloc does not exist."
    exit 1
fi
if [ ! -x "$dir/cloc" ]; then
    chmod +x "$dir/cloc"
fi

"$dir/cloc" --skip-uniqueness                                                  \
            --force-lang="CUDA",cu.disable                                     \
            --force-lang="C++",cpp.disable                                     \
            --ignored=ignored.txt                                              \
            "$dir/../include" "$dir/../src" "$dir/../test"                     \
            "$dir/../externals/cuStinger/include"                              \
            "$dir/../externals/cuStinger/externals/Support"                    \
            "$dir/../externals/cuStinger/src"                                  \
            "$dir/../externals/cuStinger/test"

if [ $? -ne 0 ]; then exit 1; fi

cd "$dir/../include/"
"$dir/cloc" --by-file .

cd "$dir/../src/"
"$dir/cloc" --by-file .

cd "$dir/../test/"
"$dir/cloc" --by-file .

cd "$dir/../externals/cuStinger/include/"
"$dir/cloc" --by-file .

cd "$dir/../externals/cuStinger/src/"
"$dir/cloc" --by-file .

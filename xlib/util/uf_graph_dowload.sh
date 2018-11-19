#!/bin/bash
function error {
   if [ $? -ne 0 ]; then
        echo "! Error: "$@
        exit $?;
    fi
}

if [ "$1" = "--help" ]; then
	echo -e "\n./uf_graph_download <directory to download> <market graph link 1> <market graph link 2> ...\n"
    exit
fi

dataset_dir=$1

for link in $@; do
    if [ $link == $1 ]; then
        continue
    fi
    wget $link -P "$dataset_dir"
    error "wget"

    file_name=`echo $link | rev | cut -d'/' -f 1 | rev`

    tar xf "$dataset_dir/$file_name" -C "$dataset_dir"
    error "tar xf" $file_name

    graph_name=${file_name::-7}
    mv "$dataset_dir/$graph_name/$graph_name.mtx" "$dataset_dir"
    error "mv" $graph_name

    rm "$dataset_dir/$file_name"
    error "rm" $graph_name

    rm -r "$dataset_dir/$graph_name"
    error "rm -r" $graph_name
done

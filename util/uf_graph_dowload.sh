#!/bin/bash
if [ "$#" -ne 2 ] || [ "$1" = "--help" ]; then
	echo -e "\n./uf_graph_download <directory_to_download> <market_graph_link> \n"
    exit
fi

dataset_dir=$1

wget $2 -P "$dataset_dir"
if [ $? -ne 0 ]; then exit 1; fi

file_name=`echo $2 | rev | cut -d'/' -f 1 | rev`

tar xf "$dataset_dir/$file_name" -C "$dataset_dir"
if [ $? -ne 0 ]; then exit 1; fi

graph_name=${file_name::-7}
mv "$dataset_dir/$graph_name/$graph_name.mtx" "$dataset_dir"
if [ $? -ne 0 ]; then exit 1; fi

rm "$dataset_dir/$file_name"
if [ $? -ne 0 ]; then exit 1; fi

rm -r "$dataset_dir/$graph_name"
if [ $? -ne 0 ]; then exit 1; fi

file_list=`ls "$1"/*.cubin | grep -vE "link|CubWrapper|GlobalSync"`

if [ $# -eq 2 ] && [ "$2" = "-cfg" ]; then
    for file in $file_list;
    do
        name=`echo "$file" | grep -oP "(.*\/)\K(.*)(?=.cubin)"`
        echo "$name".sass
        nvdisasm -c -g -sf -cfg "$file" > "$name".dot
    done

elif [ $# -eq 1 ] && [ -d "$1" ]; then
    for file in $file_list;
    do
        name=`echo "$file" | grep -oP "(.*\/)\K(.*)(?=.cubin)"`
        echo "$name".sass
        nvdisasm -c -g -sf "$file" > "$name".sass
    done

else
    echo wrong arguments: $*
    exit 1
fi

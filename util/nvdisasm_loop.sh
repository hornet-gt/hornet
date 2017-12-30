file_list=`ls "$1"/*.cubin | grep -vE "link|CubWrapper|GlobalSync"`
for file in $file_list;
do
    name=`echo "$file" | grep -oP "(.*\/)\K(.*)(?=.cubin)"`
    nvdisasm -c -g -sf "$file" > "$name".sass
done

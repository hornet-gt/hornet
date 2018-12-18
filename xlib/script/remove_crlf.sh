
#remove carriage return line feed

grep -l -r  $'\r' "$1" | xargs sed -i 's/\r//g'


valgrind --show-leak-kinds=all --leak-check=full --track-origins=yes\
         --gen-suppressions=all --log-file=valgrind.log\
         --max-stackframe=2818064 "$@"

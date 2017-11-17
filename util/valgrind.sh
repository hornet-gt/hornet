#!/bin/bash
valgrind --show-leak-kinds=all --leak-check=full --suppressions=cuda.supp \
         --error-exitcode=10 "$@"


#--track-fds=yes --track-origins=yes --max-stackframe=2818064
#valgrind --tool=drd --show-stack-usage=yes

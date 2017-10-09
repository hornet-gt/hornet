#!/bin/bash
valgrind --show-leak-kinds=all --leak-check=full --suppressions=cuda.supp "$@"

#--track-fds=yes --track-origins=yes --max-stackframe=2818064
#valgrind --tool=drd --show-stack-usage=yes

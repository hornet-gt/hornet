# Code Check #

cuda-memcheck --leak-check full --report-api-errors all

cuda-memcheck --tool synccheck

cuda-memcheck --tool initcheck

scan-build -analyze-headers --force-analyze-debug-code -maxloop 10
           -enable-checker alpha,debug,nullability,optin,security

valgrind --show-leak-kinds=all --leak-check=full --suppressions=cuda.supp

clang -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded
      -Wno-documentation-unknown-command -Wno-weak-template-vtables
      -Wno-undefined-func-template

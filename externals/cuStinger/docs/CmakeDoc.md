# Make Targets #

The CMake configuration file (util directory) provides different targets for
`make`:

*  **make** <br>
   Build the target in `RELEASE` mode. The code is optimized for maximum
   performance

*  **make update** <br>
   Clean the build directory and build the program in `RELEASE` mode.
   Disable all assertions and debugging information. The code is optimized
   for maximum performance

*  **make update_debug** <br>
   Clean the build directory and build the program in `DEBUG` mode.
   Enable all assertions and debugging information. Useful to debug the code
   with `cuda-memcheck`, `cuda-gdb`, `valgrind`, etc.

*  **make update_info** <br>
   Clean the build directory and build the program in `INFO` mode.
   Disable all assertions and debugging information but produce additional
   information in PTX assembly. Useful to analyse the code generated at
   PTX-level. It should be used combined with `make PTX`. It creates a `TMP`
   directory in the build directory with all intermediate files produced by the
   compiler

*  **make rm** <br>
   Clean the build directory. Equivalent to `rm -rf *`

*  **make PTX** <br>
   After the command `make update_info` it extracts the ptx and place it in the
   build directory (.ptx extension)

*  **make valgrind** <br>
   Copy in the `valgrind.sh` script in the build directory. The script executes
   `valgrind` with all checks enables and suppress all cuda-related warning

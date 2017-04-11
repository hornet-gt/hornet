/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
#include "Util/Parameters.hpp"
#include "Support/Parsing.hpp"     //xlib::is_integer
#include "Support/Basic.hpp"       //ERROR
#include <fstream>

namespace cu_stinger {

Param::Param(int argc, char* argv[]) : //strategy(Strategy::UNDEF),
                                       /*strategy_prop(),
                                       batch_size(0),
                                       insert(false), remove(false),*/
                                       print(false),
                                      // batch_prop(),
                                       adj_sort(false),
                                       spmv(false),
                                       binary(false) {
    if (argc == 1) {
        ERROR("Invalid number of parameters. "
               "See ", argv[0], " --help for syntax");
    }
    for (int i = 2; i < argc; i++) {
        std::string str(argv[i]);
        /*if (str == "--insert" && xlib::is_integer(argv[i + 1])) {
            insert     = true;
            batch_size = std::stoi(argv[++i]);
        }
        else if (str == "--remove" && xlib::is_integer(argv[i + 1])) {
            remove     = true;
            batch_size = std::stoi(argv[++i]);
        }
        else if (str == "--sort")
            batch_prop.sort = true;
        else if (str == "--hash-split" && xlib::is_integer(argv[i + 1])) {
            strategy_prop.hash_split = std::stoi(argv[++i]);
            batch_prop.sort          = true;
        }
        else if (str == "--spmv")
            spmv = true;*/
        if (str == "--binary")
            binary = true;
        /*else if (str == "--weighted")
            batch_prop.weighted = true;
        else if (str == "--strategy=seq")
            strategy = Strategy::SEQUENTIAL;
        else if (str == "--strategy=seq-shfl")     // CUDA
            strategy = Strategy::SEQUENTIAL_SHFL;
        else if (str == "--strategy=no-dup")
            strategy = Strategy::NO_DUP;
        else if (str == "--strategy=no-dup-shfl")  // CUDA
            strategy = Strategy::NO_DUP_SHFL;
        else if (str == "--strategy=lb-seq")
            strategy = Strategy::LB_SEQUENTIAL;
        else if (str == "--strategy=hash")
            strategy = Strategy::HASH;
        else if (str == "--strategy=multi-hash")
            strategy = Strategy::MULTI_HASH;
        else if (str == "--strategy=merge") {
            strategy = Strategy::MERGE;
            adj_sort = true;
            batch_prop.sort = true;
        }
        else if (str == "--strategy=merge-linear") {
            strategy = Strategy::MERGE_LINEAR;
            adj_sort = true;
            batch_prop.sort = true;
        }
#if defined(_OPENMP)
        else if (str == "--strategy=seq-noatomic") {
            strategy        = Strategy::SEQ_NOATOMIC;
            batch_prop.sort = true;
        }
        else if (str == "--strategy=lb-seq-noatomic") {
            strategy        = Strategy::LB_SEQ_NOATOMIC;
            batch_prop.sort = true;
        }
        else if (str == "--strategy=local-hash") {
            strategy        = Strategy::LOCAL_HASH;
            batch_prop.sort = true;
        }
#endif
        else if (str == "--vw" && xlib::is_integer(argv[i + 1])) {
            int vw_size = std::stoi(argv[++i]);
            if (!xlib::is_power2(vw_size) || vw_size < 0 || vw_size > 32)
                ERROR("vw: must be a power of 2 in the range [1, 32]")
            strategy_prop.vw_size = vw_size;
        }
        else if (str == "--hash-config=p" && xlib::is_integer(argv[i + 1])) {
            strategy_prop.hash_config    = HashConfig::PERCENTAGE;
            strategy_prop.hash_threshold = std::stoi(argv[++i]);
        }
        else if (str == "--hash-config=s" && xlib::is_integer(argv[i + 1])) {
            strategy_prop.hash_config    = HashConfig::SIZE;
            strategy_prop.hash_threshold = std::stoi(argv[++i]);
        }
        else if (str == "--hash-config=d" && xlib::is_integer(argv[i + 1])) {
            strategy_prop.hash_config    = HashConfig::DEGREE;
            strategy_prop.hash_threshold = std::stoi(argv[++i]);
        }
        else if (str == "--hash-split" && xlib::is_integer(argv[i + 1])) {
            strategy_prop.hash_split = std::stoi(argv[++i]);
            batch_prop.sort          = true;
        }
        else if (str == "--print") {
            print            = true;
            batch_prop.print = true;
        }*/
        else if (str == "--help") {
            std::ifstream syntax_file("../Syntax.txt");
            std::cout << syntax_file.rdbuf() << "\n\n";
            std::exit(EXIT_SUCCESS);
        }
        else {
            ERROR("Invalid parameter: ", str, "\n"
                  " See ", argv[0], " --help for syntax)")
        }
    }
}

} // namespace cu_stinger

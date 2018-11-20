/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Util/CommandLineParam.hpp"
#include <Device/Util/CudaUtil.cuh>     //xlib::device_info
#include <Device/Util/SafeCudaAPI.cuh>  //SAFE_CALL
#include <Host/Basic.hpp>               //ERROR, xlib::is_integer
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <fstream>                      //std::ifstream

namespace hornets_nest {

template<typename T, typename R>
CommandLineParam::CommandLineParam(graph::GraphStd<T, R>& graph,
                                   int argc, char* argv[], bool exit_if_unknown)
                                   noexcept {
    using namespace graph::parsing_prop;
    using namespace graph::structure_prop;
    graph::ParsingProp prop(PRINT_INFO);
    bool write_binary = false;
    if (argc == 1) {
L1:     std::ifstream syntax_file("../docs/Syntax.txt");
        std::cout << syntax_file.rdbuf() << std::endl;
        syntax_file.close();
        std::exit(EXIT_SUCCESS);
    }
    for (int i = 2; i < argc; i++) {
        std::string str(argv[i]);
        if (str == "--undirected")
            graph.set_structure(UNDIRECTED);
        else if (str == "--directed")
            graph.set_structure(DIRECTED);
        else if (str == "--sort-adj")
            prop += SORT;
        else if (str == "--randomize-id")
            prop += RANDOMIZE;
        else if (str == "--no-info")
            prop -= PRINT_INFO;
        else if (str == "--write-binary")
            write_binary = true;
        else if (str == "--device-info")
            xlib::device_info();
        else if (str == "--device" && xlib::is_integer(argv[i + 1]))
            cuSetDevice(std::stoi(argv[++i]));
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
        else if (str == "--help")
            goto L1;
        else if (exit_if_unknown)
            ERROR("Invalid parameter: ", str, "      see ", argv[0], " --help")
    }
    graph.read(argv[1], prop);
    if (write_binary) {
        graph.writeBinary(xlib::extract_filepath_noextension(argv[1]) + ".bin");
        std::exit(EXIT_SUCCESS);
    }
}

template CommandLineParam::CommandLineParam(graph::GraphStd<int, int>&,
                                            int, char* argv[], bool);
template CommandLineParam::CommandLineParam(graph::GraphStd<int64_t, int64_t>&,
                                            int, char* argv[], bool);

} // namespace hornets_nest

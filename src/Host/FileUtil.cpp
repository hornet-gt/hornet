/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include "Host/FileUtil.hpp"
#include "Host/Basic.hpp"   //ERROR
#include <cassert>          //assert
#include <fstream>          //std::ifstream
#include <limits>           //std::numeric_limits
#if defined(__linux__)
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

namespace xlib {

size_t file_size(const char* filename) {
     std::ifstream fin(filename);
     size_t size = file_size(fin);
     fin.close();
     return size;
}

size_t file_size(std::ifstream& fin) {
     fin.seekg(0L, std::ios::beg);
     auto start_pos = fin.tellg();
     fin.seekg(0L, std::ios::end);
     auto end_pos = fin.tellg();
     assert(end_pos > start_pos);
     fin.seekg(0L, std::ios::beg);
     return static_cast<size_t>(end_pos - start_pos);
}

void check_regular_file(std::ifstream& fin, const char* filename) {
    if (!fin.is_open() || fin.fail() || fin.bad() || fin.eof())
        ERROR("Unable to read file");
    try {
        char c;
        fin >> c;
    } catch (std::ios_base::failure&) {
        ERROR("Unable to read the file: ", filename);
    }
    struct stat info;
    if (::stat( filename, &info ) != 0)
        ERROR("Unable to read the file: ", filename)
    else if (info.st_mode & S_IFDIR)
        ERROR("The file is a directory: ", filename)
    fin.seekg(0L, std::ios::beg);
}

void check_regular_file(const char* filename) {
    std::ifstream fin(filename);
    check_regular_file(fin, filename);
    fin.close();
}

std::string extract_filename(const std::string& str) noexcept {
    auto   pos = str.find_last_of('/');
    auto start = pos == std::string::npos ? 0 : pos + 1;
    return str.substr(start, str.find_last_of('.') - start);
}

std::string extract_file_extension(const std::string& str) noexcept {
    auto last = str.find_last_of('.');
    return last == std::string::npos ? std::string() : str.substr(last);
}

std::string extract_filepath_noextension(const std::string& str) noexcept {
    return str.substr(0, str.find_last_of('.'));
}

void skip_lines(std::istream& fin, int num_lines) {
     for (int i = 0; i < num_lines; i++)
         fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void skip_words(std::istream& fin, int num_words) {
     for (int i = 0; i < num_words; i++)
         fin.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
}

} // namespace xlib

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
#include "Support/FileUtil.hpp"
#include "Support/Basic.hpp"  //ERROR
#include <cassert>              //assert
#include <fstream>              //std::ifstream
#include <limits>               //std::numeric_limits

namespace xlib {

size_t file_size(const char* filename) {
     std::ifstream fin(filename);
     size_t size = file_size(fin);
     fin.close();
     return size;
}

size_t file_size(std::ifstream& fin) {
     fin.seekg(0L, std::ios::beg);
     std::iostream::pos_type start_pos = fin.tellg();
     fin.seekg(0L, std::ios::end);
     std::iostream::pos_type end_pos = fin.tellg();
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
        ERROR("Unable to read the file ", filename);
    }
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

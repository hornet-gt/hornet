/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include <cmath>        //std::round
#include <type_traits>  //std::is_floating_point

namespace xlib {

template<typename T>
std::string format(T num, unsigned precision) noexcept {
    T round_num = !std::is_floating_point<T>::value ? num :
                  std::round(num * static_cast<T>(100)) / static_cast<T>(100);
    std::string str = std::to_string(round_num);
    auto     find_p = str.find('.');
    auto       init = find_p == std::string::npos ? str.size() : find_p;

    for (int i = static_cast<int>(init) - 3; i > 0; i -= 3)
        str.insert(static_cast<unsigned>(i), 1, ',');

    auto find_r = str.find('.');
    if (find_r != std::string::npos)
        str.erase(find_r + precision + 1);
    return str;
}

template<class T, size_t SIZE>
void printArray(T (&array)[SIZE], const std::string& str, char sep) noexcept {
    printArray(array, SIZE, str, sep);
}

template<class T>
void printArray(const T* array, size_t size, const std::string& str, char sep)
                noexcept {
    std::cout << str;
    for (size_t i = 0; i < size; i++)
        std::cout << array[i] << sep;
    std::cout << "\n" << std::endl;
}

template<>
void printArray<char>(const char* array, size_t size, const std::string& str,
                      char sep) noexcept;

template<>
void printArray<unsigned char>(const unsigned char* array, size_t size,
                               const std::string& str, char sep) noexcept;

//------------------------------------------------------------------------------

template<class T>
void printMatrix(T** matrix, int rows, int cols, const std::string& str) {
    std::cout << str;
    for (int i = 0; i < rows; i++)
        printArray(matrix[i * cols], cols, "\n", '\t');
    std::cout << "\n" << std::endl;
}

//------------------------------------------------------------------------------

namespace detail {

template<typename T>
HOST_DEVICE
typename std::enable_if<std::is_floating_point<T>::value>::type
printfArrayAux(T* array, int size) {
    for (int i = 0; i < size; i++)
        printf("%f ", array[i]);
}

template<typename T>
HOST_DEVICE
typename std::enable_if<std::is_integral<T>::value &&
                        std::is_unsigned<T>::value>::type
printfArrayAux(T* array, int size) {
    for (int i = 0; i < size; i++)
        printf("%llu ", static_cast<uint64_t>(array[i]));
}

template<typename T>
HOST_DEVICE
typename std::enable_if<std::is_integral<T>::value &&
                        std::is_signed<T>::value>::type
printfArrayAux(T* array, int size) {
    for (int i = 0; i < size; i++)
        printf("%lld ", static_cast<int64_t>(array[i]));
}

template<>
HOST_DEVICE
void printfArrayAux<char>(char* array, int size) {
    for (int i = 0; i < size; i++)
        printf("%c ", array[i]);
}

} // namespace detail

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE
void printfArray(T* array, int size) {
    detail::printfArrayAux(array, size);
    printf("\n");
}

template<typename T, int SIZE>
HOST_DEVICE
void printfArray(T (&array)[SIZE]) {
    printfArray(array, SIZE);
    printf("\n");
}

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE
void printBits(T* array, int size) {
    const auto T_SIZE = static_cast<int>( sizeof(T) * 8u );
    using R = typename std::conditional<std::is_same<T, float>::value, unsigned,
              typename std::conditional<
                std::is_same<T, double>::value, uint64_t, T>::type>::type;

    for (int i = 0; i < size; i += T_SIZE) {
        for (int j = i; j < i + T_SIZE && j < size; j++) {
            auto array_value = reinterpret_cast<R&>( array[j / T_SIZE] );
            auto        mask = static_cast<R>( 1 << (j % T_SIZE) );
            int        value = ( array_value & mask ) ? 1 : 0;
            printf("%d", value);
        }
        printf(" ");
    }
    printf("\n");
}

} // namespace xlib

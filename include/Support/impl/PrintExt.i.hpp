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
#if defined(__NVCC__)
    #include "Support/SafeFunctions.cuh"   //cuMemcpyFromSymbol
#endif
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
void printArray(T* array, size_t size, const std::string& str, char sep) noexcept {
    std::cout << str;
    for (size_t i = 0; i < size; i++)
        std::cout << array[i] << sep;
    std::cout << "\n" << std::endl;
}

template<>
void printArray<char>(char* array, size_t size, const std::string& str,
                      char sep) noexcept;

template<>
void printArray<unsigned char>(unsigned char* array, size_t size,
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
/*
#if defined(__NVCC__)

template<class T>
void printCudaArray(const T* d_array, size_t size, const std::string& str,
                    char sep) {
    using R = typename std::remove_cv<T>::type;
    auto h_array = new R[size];
    cuMemcpyToHost(d_array, h_array, size);

    printArray(h_array, size, str, sep);
    delete[] h_array;
}

template<class T>
void printCudaSymbol(const T& d_array, size_t size, const std::string& str,
                    char sep) {
    using R = typename std::remove_cv<T>::type;
    auto h_array = new R[size];
    cuMemcpyFromSymbol(d_array, h_array, size);

    printArray(h_array, size, str, sep);
    delete[] h_array;
}

template<class T, int SIZE>
void printCudaSymbol(const T (&d_array)[SIZE], const std::string& str,
                     char sep) {
    using R = typename std::remove_cv<T>::type;
    auto h_array = new R[SIZE];
    cuMemcpyFromSymbol(d_array, h_array);

    printArray(h_array, SIZE, str, sep);
    delete[] h_array;
}

#endif*/
} // namespace xlib

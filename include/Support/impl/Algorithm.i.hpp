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
#include "Support/Basic.hpp"	//ERROR
#include <algorithm>            //std::transform, std::sort
#include <cassert>              //assert
#include <thread>               //std::thread

namespace xlib {

template<typename T>
struct numeric_limits;

template<typename T, typename R>
R UniqueMap<T, R>::insertValue(T id) {
    auto it = this->find(id);
    if (it == this->end()) {
        auto node_id = static_cast<R>(this->size());
        this->insert(std::pair<T, R>(id, node_id));
        return node_id;
    }
    return it->second;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    iteratorB_t it_B = start_B;
    for (iteratorA_t it_A = start_A; it_A != end_A; it_A++, it_B++) {
        if (*it_A != *it_B) {
            if (FAULT) {
                auto dist = std::distance(start_A, it_A);
                ERROR("Array Difference at: ", std::distance(start_A, it_A),
                      " -> Left Array: ", *it_A, "     Right Array: ", *it_B);
            }
            return false;
        }
    }
    return true;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type)) {

    iteratorB_t it_B = start_B;
    for (iteratorA_t it_A = start_A; it_A != end_A; it_A++, it_B++) {
        if (!equalFunction(*it_A, *it_B)) {
            if (FAULT) {
                ERROR("Array Difference at: ", std::distance(start_A, it_A),
                      " -> Left Array: ", *it_A, "     Right Array: ", *it_B);
            }
            return false;
        }
    }
    return true;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equalSorted(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    using T = typename std::iterator_traits<iteratorA_t>::value_type;
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    auto tmpArray_A = new T[size];
    auto tmpArray_B = new R[size];

    std::copy(start_A, end_A, tmpArray_A);
    std::copy(start_B, start_B + size, tmpArray_B);
    std::sort(tmpArray_A, tmpArray_A + size);
    std::sort(tmpArray_B, tmpArray_B + size);

    bool flag = equal<FAULT>(tmpArray_A, tmpArray_A + size, tmpArray_B);

    delete[] tmpArray_A;
    delete[] tmpArray_B;
    return flag;
}

template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args) {
    if (MultiCore) {
        auto concurrency =static_cast<int>(std::thread::hardware_concurrency());
        std::thread threadArray[32];

        for (int i = 0; i < concurrency; i++)
            threadArray[i] = std::thread(FUN, Args..., i, concurrency);
        for (int i = 0; i < concurrency; i++)
            threadArray[i].join();
    } else
        FUN(Args..., 0, 1);
}

namespace detail {

template<typename S, typename R>
void sort_by_key_aux3(const S* indexes, size_t size, R* data) {
    auto tmp = new R[size];
    std::copy(data, data + size, tmp);
    std::transform(indexes, indexes + size, data,
                    [&](S index) { return tmp[index]; });
    delete[] tmp;
}

template<typename S>
void sort_by_key_aux2(const S*, size_t) {};

template<typename S, typename R, typename... RArgs>
void sort_by_key_aux2(const S* indexes, size_t size, R* data,
                      RArgs... data_packed) {
    sort_by_key_aux3(indexes, size, data);
    sort_by_key_aux2(indexes, size, data_packed...);
}

template<typename S, typename T, typename... RArgs>
void sort_by_key_aux1(T* start, T* end, RArgs... data_packed) {
    auto    size = static_cast<size_t>(std::distance(start, end));
    auto indexes = new S[size];
    std::iota(indexes, indexes + size, 0);

    auto lambda = [&](S i, S j) { return start[i] < start[j]; };
    std::sort(indexes, indexes + size, lambda);

    sort_by_key_aux3(indexes, size, start);
    sort_by_key_aux2(indexes, size, data_packed...);
    delete[] indexes;
}

} // namespace detail

/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename... RArgs>
void sort_by_key(T* start, T* end, RArgs... data_packed) {
    if (std::distance(start, end) < std::numeric_limits<int>::max())
        detail::sort_by_key_aux1<int>(start, end, data_packed...);
    else
        detail::sort_by_key_aux1<int64_t>(start, end, data_packed...);
}


/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename R>
void sort_by_key(T* start, T* end, R* data) {
    auto   size = static_cast<size_t>(std::distance(start, end));
    auto  pairs = new std::pair<T, R>[size];
    for (size_t i = 0; i < size; i++)
        pairs[i] = std::make_pair(start[i], data[i]);

    auto lambda = [](const std::pair<T, R>& a,
                     const std::pair<T, R>& b) {
                        return a.first < b.first;
                    };
    std::sort(pairs, pairs + size, lambda);
    for (size_t i = 0; i < size; i++) {
        start[i] = pairs[i].first;
        data[i]  = pairs[i].second;
    }
    delete[] pairs;
}

template<typename T, typename S>
HOST_DEVICE
void merge(const T* left, S size_left, const T* right, S size_right, T* merge) {
    S i = 0, j = 0, k = 0;
    while (i < size_left && j < size_right)
        merge[k++] = left[i] <= right[j] ? left[i++] : right[j++];

    if (i < size_left) {
        for (S p = i; p < size_left; p++)
            merge[k++] = left[p];
    }
    else {
        for (S p = j; p < size_right; p++)
            merge[k++] = right[p];
    }
}

template<typename T, typename S>
HOST_DEVICE
void inplace_merge(T* left, S size_left, const T* right, S size_right) {
    S i = size_left - 1, j = size_right - 1, k = size_left + size_right - 1;
    while (i >= 0 && j >= 0)
        left[k--] = left[i] <= right[j] ? right[j--] : left[i--];
    while(j >= 0)
        left[k--] = right[j--];
}

//==============================================================================


/**
 * The following algorithms return the leftmost place where the given element
 * can be correctly inserted (and still maintain the sorted order).
 * the lowest index where the element is equal to the given value or
 * the highest index where the element is less than the given value
 *
 * mem: {0, 3, 5, 5, 8, 8, 8, 18, 36}
 *
 * RIGHT: searched: 5    return 2
 *  LEFT: searched: 5    return 2
 *
 * RIGHT: searched: 2    return 1
 *  LEFT: searched: 2    return 0
 *
 * RIGHT: searched: -1    return 0
 *  LEFT: searched: -1    return -1
 *
 * RIGHT: searched: 40    return 9
 *  LEFT: searched: 40    return 8
 * $\Theta(log(n))$
 */
template<bool RIGHT, typename T, typename R>
HOST_DEVICE
R lower_bound(const T* mem, R size, T searched) {
    R start = 0, end = size, mid;
    bool flag = false;
    assert(size < xlib::numeric_limits<R>::max / 2 && "May overflow");

    while (start < end) {
        mid = (start + end) / 2u;// mid = low + (high - low) / 2u avoid overflow
        T tmp = mem[mid];
        if (searched <= tmp) {
            end  = mid;
            flag = searched == tmp;
        }
        else
            start = mid + 1;
    }
    return (RIGHT || flag) ? start : start - 1;
}

template<typename T, typename R>
HOST_DEVICE
R lower_bound_left(const T* mem, R size, T searched) {
    return lower_bound<false>(mem, size, searched);
}
template<typename T, typename R>
HOST_DEVICE
R lower_bound_right(const T* mem, R size, T searched) {
    return lower_bound<true>(mem, size, searched);
}

/**
 * The following algorithms return the rightmost place where the given element
 * can be correctly inserted (and still maintain the sorted order).
 *
 * mem: {0, 3, 5, 5, 8, 8, 8, 18, 36}
 *
 * RIGHT : searched: 5    return 4
 *  LEFT : searched: 5    return 3
 *
 * RIGHT : searched: 2    return 1
 *  LEFT : searched: 2    return 0
 *
 * RIGHT : searched: -1    return 0
 *  LEFT : searched: -1    return -1
 *
 * RIGHT : searched: 40    return 9
 *  LEFT : searched: 40    return 8
 */
//first greater
template<typename T, typename R>
HOST_DEVICE
R upper_bound_left(const T* mem, R size, T searched) {
    return upper_bound_right(mem, size, searched) - 1;
}

template<typename T, typename R>
HOST_DEVICE
R upper_bound_right(const T* mem, R size, T searched) {
    R start = 0, end = size, mid;
    assert(size < xlib::numeric_limits<R>::max / 2 && "May overflow");

    while (start < end) {
        mid = (start + end) / 2u; // mid = low + (high - low) / 2 avoid overflow
        if (searched >= mem[mid])
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

template<typename T, typename R>
HOST_DEVICE
R binary_search(const T* mem, R size, T searched) {
    assert(size != 0 || std::is_signed<R>::value);
    R start = 0, end = size - 1;
    while (start <= end) {
        R mid = (start + end) / 2u;
        if (mem[mid] > searched)
            end = mid - 1;
        else if (mem[mid] < searched)
            start = mid + 1;
        else
            return mid;
    }
    return size; // indicate not found
}

} // namespace xlib

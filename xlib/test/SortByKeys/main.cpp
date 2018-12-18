#include <chrono>
#include <iostream>
#include <random>
#include <XLib.hpp>


using namespace xlib;
using namespace timer2;
/*
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
    size_t  size = static_cast<size_t>(std::distance(start, end));
    auto indexes = new S[size];
    std::iota(indexes, indexes + size, 0);

    auto lambda = [&](S i, S j) { return start[i] < start[j]; };
    std::sort(indexes, indexes + size, lambda);

    sort_by_key_aux3(indexes, size, start);
    sort_by_key_aux2(indexes, size, data_packed...);
    delete[] indexes;
}
*/
/**
 * required auxilary space: O(|end -start| * 2)
 */
/*template<typename T, typename... RArgs>
void sort_by_key(T* start, T* end, RArgs... data_packed) {
    if (std::distance(start, end) < std::numeric_limits<int>::max())
        sort_by_key_aux1<int>(start, end, data_packed...);
    else
        sort_by_key_aux1<int64_t>(start, end, data_packed...);
}
*/

/**
 * required auxilary space: O(|end -start| * 2)
 *//*
template<typename T, typename R>
void sort_by_key(T* start, T* end, R* data) {
    size_t size = static_cast<size_t>(std::distance(start, end));
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
*/



const int size = 10;

int main() {
    unsigned seed = 0;// std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 10);

    auto array = new int[size];
    auto data = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = distribution(generator);
        data[i] = distribution(generator);
    }
    printArray(array, size, "array:  ");
    printArray(data, size,  "data:   ");


    Timer<HOST> TM;
    /*TM.start();

    auto pairs = new std::pair<int, int>[size];
    for (int i = 0; i < size; i++)
        pairs[i] = std::make_pair(array[i], data[i]);
    std::sort(pairs, pairs + size, [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                        return a.first < b.first;
                                    });


    TM.stop();
    std::cout << "\n" << TM.duration() << "\n\n";*/

    TM.start();

    sort_by_key(array, array + size, data);

    TM.stop();
    std::cout << "\n" << TM.duration() << "\n\n";

    printArray(array, size);
    printArray(data, size);

    delete[] array;
    delete[] data;
}

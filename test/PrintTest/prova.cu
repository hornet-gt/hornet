#include <XLib.hpp>
#include <iostream>

int main() {
    int array1[16] = {1, 2, 4, 5, 6, 7};
    float array2[10] = {0.5, 4.2};
    int64_t array3[10] = {};
    double array4[10] = {2.66, 4.22};
    char array5[10] = {'a', 'h'};

    xlib::printfArray(array1);
    xlib::printfArray(array2);
    xlib::printfArray(array3);
    xlib::printfArray(array4);
    xlib::printfArray(array5);

    xlib::printBits(array1, 16);
    xlib::printBits(array2, 16);
    xlib::printBits(array4, 16);
    xlib::printBits(array5, 16);

    int2 aaa[8];
    aaa[0] = make_int2(3, 4);
    aaa[1] = make_int2(5, 2);
    aaa[2] = make_int2(9, 3);
    aaa[3] = make_int2(3, 7);

    xlib::printArray(aaa);
}

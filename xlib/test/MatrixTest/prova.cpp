#include <XLib.hpp>
#include <iostream>

using namespace xlib;

int main() {
    Matrix<bool> A(34, 34);



    A[0][0] = 0;
    //A[2][2] = 1;

    A.print();
}

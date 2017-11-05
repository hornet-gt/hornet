#include <XLib.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include "prova.cuh"

__global__ void cci {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return 4;
    sdfs
}

int main() {
    using namespace timer2;
    //using namespace std::chrono_literals;
    Timer<HOST> TM;
    TM.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    TM.stop();
    TM.getTime();

    TM.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    TM.stop();

    TM.getTime();
    std::cout << TM.duration() << std::endl;
    std::cout << TM.total_duration() << std::endl;
}

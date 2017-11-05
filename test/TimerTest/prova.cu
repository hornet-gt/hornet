#include <XLib.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include "prova.cuh"

int main() {

    //using namespace std::chrono_literals;
    //using namespace timer;
    //Timer_cuda TM(3);

    using namespace timer2;
    Timer<DEVICE> TM(3);
    TM.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    TM.stop();
    TM.getTime();

    TM.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    TM.stop();

    TM.getTime();
    /*std::cout << TM.duration() << std::endl;
    std::cout << TM.total_duration() << std::endl;
    std::cout << TM.avg_duration() << std::endl;*/
}

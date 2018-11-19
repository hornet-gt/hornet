#include <iostream>
#include <XLib.hpp>
#include <thread>

struct A { int a, b; };

void f() {
    xlib::IosFlagSaver saver;
    std::cout << std::hex << 12 << std::endl;
    std::cout << std::setprecision(2) << 3.44444 << std::endl;
}

using namespace timer2;

int main() {
    Timer<HOST> TM;
    TM.start();

    std::this_thread::sleep_for( std::chrono::seconds(3) );

    TM.stop();
    TM.print();

    TM.start();

    std::this_thread::sleep_for( std::chrono::seconds(2) );

    TM.stop();
    TM.print();
    std::cout << "avg: " << TM.average() << " std_dev: " << TM.std_deviation()
              << std::endl;

    xlib::memInfoHost(1024 * 1024);
    std::cout << FIELD_OFFSET(A, b) << std::endl;

    std::cout << xlib::extract_filename("/root/ciao.txt") << "\n"
              << xlib::extract_filename("/root/ciao") << "\n"
              << xlib::extract_filename("ciao.txt") << "\n"
              << xlib::extract_filename("ciao") << "\n\n";

    std::cout << xlib::extract_file_extension("/root/ciao.txt") << "\n"
              << xlib::extract_file_extension("/root/ciao") << "\n\n";

    f();
    std::cout << 12 << std::endl;
    std::cout << 3.44444 << std::endl;

    std::cout << xlib::format(234223.44444) << std::endl;
}

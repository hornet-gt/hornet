/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @file
 * @brief Perform some simple tests on the MemoryManagement
 */
#include "Core/MemoryManagement.hpp"
#include <iostream>

using namespace custinger;

int main() {
    BitTree<int, 16, 512> bit_tree;
    bit_tree.statistics();

    for (int i = 0; i < 7; i++)
        bit_tree.insert();

    auto ptr1 = bit_tree.insert();
    auto ptr2 = bit_tree.insert();
    auto ptr3 = bit_tree.insert();

    bit_tree.print();

    bit_tree.remove(ptr1.second);
    bit_tree.remove(ptr2.second);
    bit_tree.remove(ptr3.second);

    bit_tree.print();

    xlib::charSequence('-');

    MemoryManagement mem_management;
    auto ptrB1 = mem_management.insert(1);
    auto ptrB2 = mem_management.insert(2);
    mem_management.insert(3);
    auto ptrB4 = mem_management.insert(4);
    mem_management.statistics();

    mem_management.remove(ptrB1.second, 1);
    mem_management.remove(ptrB2.second, 2);
    mem_management.remove(ptrB4.second, 4);
    mem_management.statistics();
}

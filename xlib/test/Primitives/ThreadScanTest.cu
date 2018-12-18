#include <iostream>
#include <chrono>
#include <functional>
#include <random>

#include "XLib.hpp"
using namespace xlib;

enum ThreadReduceOP { INCLUSIVE, INCLUSIVE_ILP, EXCLUSIVE };

__global__ void threadReduceTest(int* DataIN, int* DataOUT) {
    int Local_data[32];
    for (int i = 0; i < 32; i++)
        Local_data[i] = DataIN[i];

    ThreadReduce::Add(Local_data);
    DataOUT[SUM_OP] = Local_data[0];

    for (int i = 0; i < 32; i++)
        Local_data[i] = DataIN[i];

    ThreadReduce::Min(Local_data);
    DataOUT[MIN_OP] = Local_data[0];

    for (int i = 0; i < 32; i++)
        Local_data[i] = DataIN[i];

    ThreadReduce::Max(Local_data);
    DataOUT[MAX_OP] = Local_data[0];

    for (int i = 0; i < 32; i++)
        Local_data[i] = DataIN[i];

    ThreadReduce::LogicAnd(Local_data);
    DataOUT[LOGIC_AND_OP] = Local_data[0];
}


int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(-50, 50);

    const int INPUT_SIZE = 32;
    const int N_OF_OPERATIONS = 4;
    int DataIN[INPUT_SIZE];
    int DataOUT[N_OF_OPERATIONS];
    int DataOUT_copy[N_OF_OPERATIONS];
    int* devDataIN, *devDataOUT;
    __SAFE_CALL( cudaMalloc(&devDataIN, sizeof(DataIN)) );
    __SAFE_CALL( cudaMalloc(&devDataOUT, sizeof(DataOUT)) );

    for (int i = 0; i < INPUT_SIZE; i++)
        DataIN[i] = distribution(generator);

    xlib::printArray(DataIN, 32);

    __SAFE_CALL( cudaMemcpy(devDataIN, DataIN, sizeof(DataIN),
                 cudaMemcpyHostToDevice) );

    threadReduceTest<<<1, 1>>>(devDataIN, devDataOUT);

    __SAFE_CALL( cudaMemcpy(DataOUT_copy, devDataOUT, sizeof(DataOUT),
                 cudaMemcpyDeviceToHost) );

    DataOUT[SUM_OP] = std::accumulate(DataIN, DataIN + INPUT_SIZE, 0);
    if (DataOUT[SUM_OP] != DataOUT_copy[SUM_OP]) {
        ERROR("ThreadReduce (SUM) : " << DataOUT[SUM_OP] << "\t"
                                        << DataOUT_copy[SUM_OP]);
    }

    DataOUT[MIN_OP] = *std::min_element(DataIN, DataIN + INPUT_SIZE);
    if (DataOUT[MIN_OP] != DataOUT_copy[MIN_OP]) {
        ERROR("ThreadReduce (Min) : " << DataOUT[MIN_OP] << "\t"
                                        << DataOUT_copy[MIN_OP]);
    }

    DataOUT[MAX_OP] = *std::max_element(DataIN, DataIN + INPUT_SIZE);
    if (DataOUT[MAX_OP] != DataOUT_copy[MAX_OP]) {
        ERROR("ThreadReduce (Max) : " << DataOUT[MAX_OP] << "\t"
                                        << DataOUT_copy[MAX_OP]);
    }

    DataOUT[LOGIC_AND_OP] = DataIN[0];
    for (int i = 1; i < INPUT_SIZE; i++)
        DataOUT[LOGIC_AND_OP] = DataOUT[LOGIC_AND_OP] && DataIN[i];

    if (DataOUT[LOGIC_AND_OP] != DataOUT_copy[LOGIC_AND_OP]) {
        ERROR("ThreadReduce (AND) : " << DataOUT[LOGIC_AND_OP] << "\t"
                                        << DataOUT_copy[LOGIC_AND_OP]);
    }
}

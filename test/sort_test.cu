#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <Device/Util/Timer.cuh>             //xlib::Timer
#include <string>
#include <algorithm>                    //std:.generate
#include <random>
#include <thrust/device_vector.h>                    //std:.generate

//// edges sorted by src
//    const int karate_src[] = {
//// 0, 0, 0, 0, 1, 1, 9, 16
//// 0, 0, 1, 2, 2, 2, 3, 3, 4
//1, 2, 2, 3, 4, 4, 5, 5, 5
//    };
//    const int karate_dst[] = {
//// 3, 5, 8, 19, 17, 21, 2, 6
//// 1, 2, 2, 3, 4, 5, 4, 5, 5
//0, 0, 1, 2, 2, 3, 2, 3, 4
//    };
//    const int karate_batch_size = 9;

void exec(int argc, char* argv[]);
void gen_rand(const int batch_size, thrust::device_vector<int> &fst, thrust::device_vector<int> &scd);

int main(int argc, char* argv[]) {
    exec(argc, argv);
    cudaDeviceReset();
    return 0;
}

void exec(int argc, char* argv[]) {
    int batch_size = std::stoi(argv[1]);
    thrust::device_vector<int> fst;
    thrust::device_vector<int> scd;
    gen_rand(batch_size, fst, scd);
    thrust::device_vector<int> tmp_fst(batch_size);
    thrust::device_vector<int> tmp_scd(batch_size);

    xlib::CubSortPairs2<int, int> cub_sort_pair;

    cub_sort_pair.initialize(batch_size, false);
    xlib::gpu::printArray(thrust::raw_pointer_cast(fst.data()), batch_size, "(Before):\n");
    xlib::gpu::printArray(thrust::raw_pointer_cast(scd.data()), batch_size);
    cub_sort_pair.run(
            thrust::raw_pointer_cast(fst.data()),
            thrust::raw_pointer_cast(scd.data()),
            batch_size,
            thrust::raw_pointer_cast(tmp_fst.data()),
            thrust::raw_pointer_cast(tmp_scd.data()),
            batch_size, batch_size);
    xlib::gpu::printArray(thrust::raw_pointer_cast(fst.data()), batch_size, "(After):\n");
    xlib::gpu::printArray(thrust::raw_pointer_cast(scd.data()), batch_size);
}

void gen_rand(const int batch_size, thrust::device_vector<int> &fst, thrust::device_vector<int> &scd) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    std::vector<int> tmp_fst;
    std::vector<int> tmp_scd;
    for (int i = 0; i < batch_size; ++i) {
        tmp_fst.push_back(dis(gen));
    }
    std::partial_sum(tmp_fst.begin(), tmp_fst.end(), tmp_fst.begin());
    std::shuffle(tmp_fst.begin(), tmp_fst.end(), gen);
    for (int i = 0; i < batch_size; ++i) {
        tmp_scd.push_back(i);
    }
    thrust::device_vector<int> tmp_dev_fst = tmp_fst;
    thrust::device_vector<int> tmp_dev_scd = tmp_scd;
    fst.swap(tmp_dev_fst);
    scd.swap(tmp_dev_scd);
}

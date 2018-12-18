

__device__ int2 Adj_array[5];

__device__ int F_array[64];

__device__ int Ouptput[64];



__global__ void prova2() {

    const int globalID = 32 * blockIdx.x + threadIdx.x;

    for (int i = globalID; i < 64; i += gridDim.x * 32)

        Ouptput[threadIdx.x] = F_array[i];

}



int main() {

    int input[64];

    for (int i = 0; i < 64; i++)

        input[i] = i;

    cudaMemcpyToSymbol(F_array, input, 64 * sizeof(int));



    int2 adj_input[64];

    adj_input[0] = make_int2(0, 3);

    adj_input[1] = make_int2(3, 7);

    adj_input[2] = make_int2(7, 10);

    adj_input[3] = make_int2(10, 12);

    adj_input[4] = make_int2(12, 15);

    cudaMemcpyToSymbol(Adj_array, adj_input, 5 * sizeof(int2));



    prova2<<<1, 32>>>();

    //prova2<<<1, 32>>>();

}




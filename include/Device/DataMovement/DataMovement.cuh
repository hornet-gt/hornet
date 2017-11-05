



template<int NUM_LOOPS, unsigned BLOCK_SIZE, typename T>
void stride_load(const T* __restrict__ input,
                 int                   num_items,
                 T*       __restrict__ output) {
    #pragma unroll
    for (int i = 0; i < NUM_LOOPS * BLOCK_SIZE; i += BLOCK_SIZE)
        output[i] = input[i];
    if (threadIdx.x < (NUM_LOOPS == 0 ? num_items : num_items % BLOCK_SIZE))
        output[NUM_LOOPS * BLOCK_SIZE] = input[NUM_LOOPS * BLOCK_SIZE];
}

template<unsigned BLOCK_SIZE, typename T>
void stride_load(const T* __restrict__ input,
                 int                   num_items,
                 T*       __restrict__ output) {

    int              num_loop = num_items / BLOCK_SIZE;
    const auto&  input_stride = input + threadIdx.x;
    const auto& output_stride = input + threadIdx.x;
    switch (num_loop) {
        case 0: stride_load<0, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 1: stride_load<1, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 2: stride_load<2, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 3: stride_load<3, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 4: stride_load<4, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 5: stride_load<5, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 6: stride_load<6, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 7: stride_load<7, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 8: stride_load<8, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 9: stride_load<9, BLOCK_SIZE>(input_stride, num_items,
                                           output_stride);
                break;
        case 10: stride_load<10, BLOCK_SIZE>(input_stride, num_items,
                                             output_stride);
                break;
        case 11: stride_load<11, BLOCK_SIZE>(input_stride, num_items,
                                             output_stride);
                break;
        case 12: stride_load<12, BLOCK_SIZE>(input_stride, num_items,
                                             output_stride);
                break;
        case 13: stride_load<13, BLOCK_SIZE>(input_stride, num_items,
                                       output_stride);
                break;
        case 14: stride_load<14, BLOCK_SIZE>(input_stride, num_items,
                                       output_stride);
            break;
        case 15: stride_load<15, BLOCK_SIZE>(input_stride, num_items,
                                       output_stride);
                break;
        default:
            assert(false);
            *static_cast<int*>(nullptr) = 0;

    }
}

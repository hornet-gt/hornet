
constexpr unsigned maskArray(unsigned lane) {
    return 1 << (lane + 1)) - 1;
}

typename xlib::GenerateSeq<maskArray, 31>::type MASK_ARRAY;

unsigned lane_mask(unsigned lane) {
    switch (lane) {
        case  0: return 0b1;
        case  1: return 0b11;
        case  2: return 0b111;
        case  3: return 0b1111;
        case  4: return 0b11111;
        case  5: return 0b111111;
        case  6: return 0b1111111;
        case  7: return 0b11111111;
        case  8: return 0b111111111;
        case  9: return 0b1111111111;
        case 10: return 0b11111111111;
        case 11: return 0b111111111111;
        case 12: return 0b1111111111111;
        case 13: return 0b11111111111111;
        case 14: return 0b111111111111111;
        case 15: return 0b1111111111111111;
        case 16: return 0b11111111111111111;
        case 17: return 0b111111111111111111;
        case 18: return 0b1111111111111111111;
        case 19: return 0b11111111111111111111;
        case 20: return 0b111111111111111111111;
        case 21: return 0b1111111111111111111111;
        case 22: return 0b11111111111111111111111;
        case 23: return 0b111111111111111111111111;
        case 24: return 0b1111111111111111111111111;
        case 25: return 0b11111111111111111111111111;
        case 26: return 0b111111111111111111111111111;
        case 27: return 0b1111111111111111111111111111;
        case 28: return 0b11111111111111111111111111111;
        case 29: return 0b111111111111111111111111111111;
        case 30: return 0b1111111111111111111111111111111;
        case 31: return 0b11111111111111111111111111111111;
    }
}


bool is_outsize(unsigned dest, unsigned mask) {
    return LaneMaskGT() & mask_fun(dest) & mask;
}

template<typename T>
T warp_segmented_reduce(T value, unsigned mask) {
    auto lane_id = lane_id()
    #pragma unroll
    for (int i = 1; i < xlib::WARP_SIZE; i *= 2) {
        auto dest = lane_id + i;
        auto  tmp = __shfl(value, dest);
        if (!is_outsize(dest, mask))
            value += tmp;
    }
    return value;
}

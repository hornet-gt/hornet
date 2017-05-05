#pragma once

#include "Core/cuStinger.hpp"                    //custinger::cuStinger
#include "Core/LoadBalancing/BinarySearch.cuh"   //load_balacing::BinarySearch

namespace custinger_alg {

//Static Algorithms Abstract class
class StaticAlgorithm {
public:
    StaticAlgorithm(const custinger::cuStinger& custinger_) noexcept :
                    custinger(custinger_),
                    load_balacing(custinger_.csr_offsets(), custinger_.nV()) {}

    virtual ~StaticAlgorithm() noexcept = 0;

    virtual void run()      = 0;
    virtual void reset()    = 0;
    virtual void release()  = 0;
    virtual bool validate() = 0;

    template<typename T>
    void register_data(const T& data) noexcept;

    virtual void syncDeviceWithHost() noexcept final;
    virtual void syncHostWithDevice() noexcept final;

protected:
    load_balacing::BinarySearch load_balacing;
    const custinger::cuStinger& custinger;

private:
    size_t _data_size     { 0 };
    void*  _h_ptr         { nullptr };
    void*  _d_ptr         { nullptr };
    bool   _is_registered { false };
};

//==============================================================================

class Allocate {
public:
    template<typename T>
    explicit Allocate(T*& pointer, size_t num_items) noexcept;

    ~Allocate() noexcept;
private:
    void* _pointer;
};

} // namespace custinger_alg

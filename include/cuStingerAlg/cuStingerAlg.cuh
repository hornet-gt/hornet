#pragma once

#include "Core/cuStinger.hpp"   //cu_stinger::cuStinger

namespace cu_stinger_alg {

//Static Algorithms Abstract class
class StaticAlgorithm {
public:
    StaticAlgorithm(const cu_stinger::cuStinger& custinger) :
                    _custinger(custinger) {}
    virtual ~StaticAlgorithm() = 0;

    virtual void run()      = 0;
    virtual void reset()    = 0;
    virtual void release()  = 0;
    virtual bool validate() = 0;
protected:
    const cu_stinger::cuStinger& _custinger;
};

//==============================================================================

class Allocate {
public:
    template<typename T>
    Allocate(T*& pointer, size_t num_items) noexcept;
    ~Allocate() noexcept;
private:
    void* _pointer;
};

template<typename T>
Allocate::Allocate(T*& pointer, size_t num_items) noexcept {
    //SAFE_CALL( cudaMallocManaged(&pointer, num_items * sizeof(T)) )
    cuMalloc(pointer, num_items);
    _pointer = pointer;
}

inline Allocate::~Allocate() noexcept {
    cuFree(_pointer);
}

} // namespace cu_stinger_alg

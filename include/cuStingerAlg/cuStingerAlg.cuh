#pragma once

#include "Core/cuStinger.hpp"   //custinger::cuStinger

namespace custinger_alg {

//Static Algorithms Abstract class
class StaticAlgorithm {
public:
    StaticAlgorithm(const custinger::cuStinger& custinger) :
                    _custinger(custinger) {}

    virtual ~StaticAlgorithm() = 0;

    virtual void run()      = 0;
    virtual void reset()    = 0;
    virtual void release()  = 0;
    virtual bool validate() = 0;
protected:
    const custinger::cuStinger& _custinger;
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

template<typename T>
Allocate::Allocate(T*& pointer, size_t num_items) noexcept {
    cuMalloc(pointer, num_items);
    _pointer = pointer;
}

inline Allocate::~Allocate() noexcept {
    cuFree(_pointer);
}

} // namespace custinger_alg

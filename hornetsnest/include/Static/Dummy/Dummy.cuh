#ifndef DUMMY_CUH
#define DUMMY_CUH

#include <HornetAlg.hpp>

namespace hornets_nest {

using vert_t = int;
using eoff_t = int;

using HornetGraph       = ::hornet::gpu::Hornet<vert_t>;
using HornetStaticGraph = ::hornet::gpu::HornetStatic<vert_t>;
using HornetInit        = ::hornet::HornetInit<vert_t>;

class Dummy : StaticAlgorithm<HornetGraph>{
    public:
    Dummy(HornetGraph& h);
    ~Dummy();
    void reset()    ;
    void run()      ;
    void release()  ;
    bool validate() ;
};

class DummyStatic : StaticAlgorithm<HornetStaticGraph>{
    public:
    DummyStatic(HornetStaticGraph& h);
    ~DummyStatic();
    void reset()    ;
    void run()      ;
    void release()  ;
    bool validate() ;
};

}
#endif

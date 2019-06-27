#include "Static/Dummy/Dummy.cuh"

namespace hornets_nest {
    Dummy::Dummy(HornetGraph& h) :
        StaticAlgorithm(h) {}

    Dummy::~Dummy() {}

    void Dummy::reset() {}

    void Dummy::run() {}

    void Dummy::release() {}

    bool Dummy::validate() {return true;}

    DummyStatic::DummyStatic(HornetStaticGraph& h) :
        StaticAlgorithm(h) {}

    DummyStatic::~DummyStatic() {}

    void DummyStatic::reset() {}

    void DummyStatic::run() {}

    void DummyStatic::release() {}

    bool DummyStatic::validate() {return true;}

}

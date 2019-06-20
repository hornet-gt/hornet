#ifndef HORNETSTATIC_CUH
#define HORNETSTATIC_CUH

#include "../Conf/Common.cuh"
#include "../Conf/HornetConf.cuh"
#include "../HornetDevice/HornetDevice.cuh"
#include "../HornetInitialize/HornetInit.cuh"

namespace hornet {
namespace gpu {

template <typename, typename = EMPTY,
         typename = EMPTY, typename = DEGREE_T>
         class HornetStatic;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
class HornetStatic<
    vid_t,
    TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>,
    degree_t> {

public:

    using HornetDeviceT = hornet::HornetDevice<TypeList<VertexMetaTypes...>, TypeList<EdgeMetaTypes...>, vid_t, degree_t>;
    using EdgeAccessT = TypeList<degree_t, xlib::byte_t*, degree_t, degree_t>;
    using HInitT = hornet::HornetInit<
        vid_t,
        TypeList<VertexMetaTypes...>,
        TypeList<EdgeMetaTypes...>, degree_t>;
    using VertexTypes = TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>;

    using VertexType = vid_t;

    using DegreeType = degree_t;

private:

    static int _instance_count;

    vid_t    _nV { 0 };
    degree_t _nE { 0 };
    int      _id { 0 };

    SoAData<
        TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
        DeviceType::DEVICE> _vertex_data;

    CSoAData<
        TypeList<vid_t, EdgeMetaTypes...>,
        DeviceType::DEVICE> _edge_data;

    void initialize(HInitT& h_init) noexcept;

public:

    HornetStatic(HInitT& h_init) noexcept;

    void print(void);

    degree_t nV(void) const noexcept;

    degree_t nE(void) const noexcept;

    HornetDeviceT device(void) noexcept;
};

#define HORNETSTATIC HornetStatic<vid_t,\
                      TypeList<VertexMetaTypes...>,\
                      TypeList<EdgeMetaTypes...>,\
                      degree_t>
}


template<typename>
class IsHornetStatic : public std::false_type {};

template<typename V, typename VM, typename EM, typename D>
class IsHornetStatic<gpu::HornetStatic<V, VM, EM, D>> : public std::true_type {};

}

#include "Core/HornetInitialize/HornetStaticInitialize.i.cuh"
#endif

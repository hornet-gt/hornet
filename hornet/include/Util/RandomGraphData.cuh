#ifndef RANDOM_GRAPH_DATA
#define RANDOM_GRAPH_DATA

//#include <Hornet.hpp>
#include "../Core/SoA/SoAData.cuh"
#include "../Core/Static/Static.cuh"
#include <random>                       //std::mt19937_64
#include <utility>                       //std::mt19937_64
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace hornet {

template <typename T>
struct RandomValues {
    //Depending upon the type T, select
    //real or integer distribution
    using Distribution = typename
    std::conditional<
    std::is_integral<T>::value,
    typename std::uniform_int_distribution<T>,
    typename std::uniform_real_distribution<T>>::type;

    //Minimum value of the distribution
    T min;

    //Maximum value of the distribution
    T max;

    //Random device
    mutable std::random_device rd;

    //Mersenne Twister generator
    mutable std::mt19937 gen;

    //Object of selected distribution type
    mutable Distribution dis;

    //Constructor to set minimum and maximum values of the distribution
    RandomValues(const T _min, const T _max) :
        min(_min), max(_max), gen(rd()), dis(min, max) {}

    //Constructor to set minimum and maximum values of the distribution
    RandomValues(const T _min, const T _max, size_t seed) :
        min(_min), max(_max), gen(seed), dis(min, max) {}

    //Operator to generate random value (const variant)
    T operator()(void) const { return dis(gen); }

    //Operator to generate random value
    T operator()(void) { return dis(gen); }
};

template <typename V, typename... T>
void
randomizeVertices(
    SoAPtr<V, V, T...> ptr,
    const graph::GraphStd<>& graph, int& batch_size,
    const bool insert, const bool unique, size_t seed = 0) {
  V* sPtr = ptr.template get<0>();
  V* dPtr = ptr.template get<1>();
  if (!insert) {
    std::mt19937_64 gen(seed);
    RandomValues<V> rSrc(0, graph.nV() - 1, seed);
    for (int i = 0; i < batch_size; i++) {
      auto src = rSrc();
      if (graph.out_degree(src) == 0) { --i; continue; }
      RandomValues<V> rDst(0, graph.out_degree(src) - 1, seed);
      auto index = rDst();
      sPtr[i] = src;
      dPtr[i] = graph.vertex(src).neighbor_id(index);
    }
  } else {
    std::mt19937_64 gen(seed);
    RandomValues<V> rVtx(0, graph.nV() - 1, seed);
    for (int i = 0; i < batch_size; i++) {
      sPtr[i] = rVtx();
      dPtr[i] = rVtx();
    }
  }

  if (unique) {
    auto temp = new std::pair<V, V>[batch_size];
    for (int i = 0; i < batch_size; i++)
      temp[i] = std::make_pair(sPtr[i], dPtr[i]);

    std::sort(temp, temp + batch_size);
    auto it = std::unique(temp, temp + batch_size);
    batch_size = std::distance(temp, it);
    for (int i = 0; i < batch_size; i++) {
      sPtr[i] = temp[i].first;
      dPtr[i] = temp[i].second;
    }
  }
}

template <typename = EMPTY> struct DefaultMinMax;

template <typename... T>
struct DefaultMinMax<TypeList<T...>> {

  template <unsigned N = 0>
  static
  typename std::enable_if<(N < sizeof...(T)), void>::type
  set(std::tuple<std::pair<T, T>...>& t) {
    using Type = typename xlib::SelectType<N, T...>::type;
    std::get<N>(t) = std::make_pair(std::numeric_limits<Type>::lowest(), std::numeric_limits<Type>::max());
    set<N+1>(t);
  }

  template <unsigned N = 0>
  static
  typename std::enable_if<(N == sizeof...(T)), void>::type
  set(std::tuple<std::pair<T, T>...>& t) { }

  static
  std::tuple<std::pair<T, T>...>
  assign(void) {
    std::tuple<std::pair<T, T>...> t;
    set<0>(t);
    return t;
  }
};

template <typename = EMPTY> struct RandomGenTraits;

template <typename... T>
struct RandomGenTraits<TypeList<T...>> {
  bool insert;
  bool unique;
  size_t seed;
  std::tuple<std::pair<T, T>...> minMax;
  RandomGenTraits(
      bool ins = true,
      bool uni = false,
      size_t s = 0,
      std::tuple<std::pair<T, T>...> mm = DefaultMinMax<TypeList<T...>>::assign()) :
    insert(ins), unique(uni), seed(s), minMax(mm) {}
};

template <typename T>
struct RFunc {
  T lb;
  T ub;
  RFunc(T l, T u) : lb(l), ub(u) {}
  using Dist = typename
    std::conditional<
    std::is_integral<T>::value,
    thrust::uniform_int_distribution<T>,
    thrust::uniform_real_distribution<T>>::type;

  __host__ __device__
    T operator() (T index) { //okay
      thrust::minstd_rand rng(index);
      Dist dist(lb, ub);
      rng.discard(index);
      return dist(rng);
    };
};

template <DeviceType device_t, typename T>
using Vector = typename
std::conditional<
(device_t == DeviceType::DEVICE),
typename thrust::device_vector<T>,
typename thrust::host_vector<T>>::type;

template <typename... EdgeMetaTypes, typename vid_t, typename degree_t, DeviceType device_t>
COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>
selectRandom(
    COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t>& coo,
    degree_t size,
    RandomGenTraits<TypeList<EdgeMetaTypes...>> t) {

  Vector<device_t, degree_t> indexKey(coo.size());
  Vector<device_t, degree_t> indexVal(coo.size());
  thrust::counting_iterator<degree_t> index_sequence_begin(t.seed);
  thrust::transform(index_sequence_begin,
      index_sequence_begin + indexKey.size(),
      indexKey.begin(), RFunc<degree_t>(0, coo.size() - 1));

  thrust::sequence(indexVal.begin(), indexVal.end());
  thrust::sort_by_key(indexKey.begin(), indexKey.end(), indexVal.begin());

  COO<device_t, vid_t, TypeList<EdgeMetaTypes...>, degree_t> selectCOO(size);
  indexVal.resize(size);
  selectCOO.gather(coo, indexVal);
  return selectCOO;
}

template <unsigned N, unsigned SIZE>
struct RecursiveRandom {
  template <typename V, typename... T>
  static void assign(
    SoAPtr<V, V, T...> ptr, int batch_size,
    std::tuple<std::pair<T, T>...>& minMax, size_t seed = 0) {
    using M = typename xlib::SelectType<N, T...>::type;
    thrust::device_ptr<M> metaPtr(ptr.template get<N + 2>());
    thrust::counting_iterator<size_t> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin,
        index_sequence_begin + batch_size,
        metaPtr, RFunc<M>(std::get<N>(minMax).first, std::get<N>(minMax).second));
    RecursiveRandom<N+1, SIZE>::assign(ptr, batch_size, minMax, seed);
  }
};

template <unsigned N>
struct RecursiveRandom<N, N> {
  template <typename V, typename... T>
  static void assign(
    SoAPtr<V, V, T...> ptr, int batch_size,
    std::tuple<std::pair<T, T>...>& minMax, size_t seed = 0) {
    using M = typename xlib::SelectType<N, T...>::type;
    thrust::device_ptr<M> metaPtr(ptr.template get<N + 2>());
    thrust::counting_iterator<size_t> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin,
        index_sequence_begin + batch_size,
        metaPtr, RFunc<M>(std::get<N>(minMax).first, std::get<N>(minMax).second));
  }
};

template <typename V, typename... T>
void
randomizeEdgeMeta(
    SoAPtr<V, V, T...> ptr, int batch_size,
    std::tuple<std::pair<T, T>...>& minMax, size_t seed = 0) {
  RecursiveRandom<0, sizeof...(T) - 1>::assign(ptr, batch_size, minMax, seed);
}

template <typename vid_t, typename degree_t, typename... EdgeMetaTypes>
typename std::enable_if<(0 == sizeof...(EdgeMetaTypes)),
COO<DeviceType::DEVICE, vid_t, TypeList<EdgeMetaTypes...>, degree_t>>::type
generateRandomCOO(vid_t nV, degree_t size, RandomGenTraits<TypeList<EdgeMetaTypes...>> t) {
  COO<DeviceType::DEVICE, vid_t, TypeList<EdgeMetaTypes...>, degree_t> coo(size);
  thrust::device_ptr<vid_t> src(coo.srcPtr());
  thrust::device_ptr<vid_t> dst(coo.dstPtr());
  thrust::counting_iterator<degree_t> index_sequence_begin(t.seed);
  thrust::transform(index_sequence_begin,
      index_sequence_begin + size,
      src, RFunc<degree_t>(0, nV - 1));
  thrust::transform(index_sequence_begin,
      index_sequence_begin + size,
      dst, RFunc<degree_t>(0, nV - 1));
  return coo;
}

template <typename vid_t, typename degree_t, typename... EdgeMetaTypes>
typename std::enable_if<(0 < sizeof...(EdgeMetaTypes)),
COO<DeviceType::DEVICE, vid_t, TypeList<EdgeMetaTypes...>, degree_t>>::type
generateRandomCOO(vid_t nV, degree_t size, RandomGenTraits<TypeList<EdgeMetaTypes...>> t) {
  COO<DeviceType::DEVICE, vid_t, TypeList<EdgeMetaTypes...>, degree_t> coo(size);
  thrust::device_ptr<vid_t> src(coo.srcPtr());
  thrust::device_ptr<vid_t> dst(coo.dstPtr());
  thrust::counting_iterator<degree_t> index_sequence_begin(t.seed);
  thrust::transform(index_sequence_begin,
      index_sequence_begin + size,
      src, RFunc<degree_t>(0, nV - 1));
  thrust::transform(index_sequence_begin,
      index_sequence_begin + size,
      dst, RFunc<degree_t>(0, nV - 1));
  randomizeEdgeMeta(coo.getPtr(), size, t.minMax, t.seed);
  return coo;
}

}

#endif

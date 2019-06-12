#ifndef RANDOM_GRAPH_DATA
#define RANDOM_GRAPH_DATA

//#include <Hornet.hpp>
#include "../Core/SoA/SoAData.cuh"
#include <random>                       //std::mt19937_64

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

template <unsigned N, unsigned SIZE>
struct RecursiveRandom {
  template <typename V, typename... T>
  static void assign(
    SoAPtr<V, V, T...> ptr, int batch_size,
    std::tuple<std::pair<T, T>...>& minMax, size_t seed = 0) {
    using M = typename xlib::SelectType<N, T...>::type;
    M metaPtr = ptr.template get<N + 2>();
    RandomValues<M> rMeta(std::get<N>(minMax).first, std::get<N>(minMax).second, seed);
    for (int i = 0; i < batch_size; ++i) {
      metaPtr[i] = rMeta();
    }
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
    M metaPtr = ptr.template get<N + 2>();
    RandomValues<M> rMeta(std::get<N>(minMax).first, std::get<N>(minMax).second, seed);
    for (int i = 0; i < batch_size; ++i) {
      metaPtr[i] = rMeta();
    }
  }
};

template <typename V, typename... T>
void
randomizeEdgeMeta(
    SoAPtr<V, V, T...> ptr, int& batch_size,
    std::tuple<std::pair<T, T>...>& minMax, size_t seed = 0) {
  RecursiveRandom<0, sizeof...(T) - 1>::assign(ptr, batch_size, minMax, seed);
}

template <typename T>
SoAData<TypeList<T, T>, DeviceType::HOST>
generateBatchData(const graph::GraphStd<>& graph, int& batch_size,
              const bool insert = true, const bool unique = false, size_t seed = 0) {
  SoAData<TypeList<T, T>, DeviceType::HOST> batchData(batch_size);
  randomizeVertices(batchData.get_soa_ptr(), graph, batch_size, insert, unique, seed);
  return batchData;
}

template <typename V, typename... T>
typename std::enable_if<
  (0 < sizeof...(T)),
  SoAData<TypeList<V, V, T...>, DeviceType::HOST>>::type
generateBatchData(const graph::GraphStd<>& graph, int& batch_size,
              std::tuple<std::pair<T, T>...>& minMax,
              const bool insert = true, const bool unique = false, size_t seed = 0) {
  SoAData<TypeList<V, V, T...>, DeviceType::HOST> batchData(batch_size);
  randomizeVertices(batchData.get_soa_ptr(), graph, batch_size, insert, unique, seed);
  randomizeEdgeMeta(batchData.get_soa_ptr(), batch_size, minMax, seed);
  
  return batchData;
}

}

#endif

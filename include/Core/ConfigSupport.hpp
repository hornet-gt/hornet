#pragma once

#include <tuple>
#include "Support/Metaprogramming.hpp"

template<typename... TArgs>
using TypeList = std::tuple<TArgs...>;

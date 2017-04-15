#pragma once

#include <tuple>

template<typename... TArgs>
using TypeList = std::tuple<TArgs...>;

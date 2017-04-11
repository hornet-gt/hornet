/*------------------------------------------------------------------------------
Copyright © 2017 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/*
 * @author Federico Busato
 *         Univerity of Verona, Dept. of Computer Science
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 */
namespace graph {

inline Property::Property(int state) noexcept : _state(state) {}

inline bool Property::is_undefined() const noexcept {
    return _state == 0;
}

inline bool Property::is_sort() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::SORT));
}

inline bool Property::is_randomize() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::RANDOMIZE));
}

inline bool Property::is_print() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::PRINT));
}
//------------------------------------------------------------------------------

inline Structure::Structure(int state) noexcept : _state(state) {}

inline bool Structure::is_undefined() const noexcept {
    return _state == 0;
}

inline bool Structure::is_directed() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::DIRECTED));
}

inline bool Structure::is_undirected() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::UNDIRECTED));
}

inline bool Structure::is_reverse() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::REVERSE));
}

inline bool Structure::is_coo() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::COO));
}

inline bool Structure::is_direction_set() const noexcept {
    return static_cast<bool>(_state & static_cast<int>(Enum::DIRECTED)) ||
           static_cast<bool>(_state & static_cast<int>(Enum::UNDIRECTED));
}

inline void Structure::operator|=(int value) noexcept {
    _state &= ~3;   //clear DIRECTED/UNDIRECTED
    _state |= value;
    /*if ((is_directed() && value == static_cast<int>(Enum::UNDIRECTED)) ||
        (is_undirected() && value == static_cast<int>(Enum::DIRECTED)))
        ERROR("Structure DIRECTED and UNDIRECTED not allowed");*/
}
//------------------------------------------------------------------------------

template<typename id_t, typename off_t>
inline GraphBase<id_t, off_t>::GraphBase(Structure structure) noexcept :
                                                    _structure(structure) {}

template<typename id_t, typename off_t>
inline GraphBase<id_t, off_t>::~GraphBase() noexcept {}

template<typename id_t, typename off_t>
inline id_t GraphBase<id_t, off_t>::nV() const noexcept {
    return _V;
}

template<typename id_t, typename off_t>
inline off_t GraphBase<id_t, off_t>::nE() const noexcept {
    return _E;
}

template<typename id_t, typename off_t>
inline const std::string& GraphBase<id_t, off_t>::name() const noexcept {
    return _graph_name;
}

} // namespace graph

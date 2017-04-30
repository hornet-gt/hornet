/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
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

inline bool Structure::is_weighted() const noexcept {
    return _wtype != NONE;
}

inline void Structure::operator|=(int value) noexcept {
    _state &= ~3;   //clear DIRECTED/UNDIRECTED
    _state |= value;
    /*if ((is_directed() && value == static_cast<int>(Enum::UNDIRECTED)) ||
        (is_undirected() && value == static_cast<int>(Enum::DIRECTED)))
        ERROR("Structure DIRECTED and UNDIRECTED not allowed");*/
}
//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
inline GraphBase<vid_t, eoff_t>::GraphBase(Structure structure) noexcept :
                                                    _structure(structure) {}

template<typename vid_t, typename eoff_t>
inline vid_t GraphBase<vid_t, eoff_t>::nV() const noexcept {
    return _nV;
}

template<typename vid_t, typename eoff_t>
inline eoff_t GraphBase<vid_t, eoff_t>::nE() const noexcept {
    return _nE;
}

template<typename vid_t, typename eoff_t>
inline const std::string& GraphBase<vid_t, eoff_t>::name() const noexcept {
    return _graph_name;
}

} // namespace graph

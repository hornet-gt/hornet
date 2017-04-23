/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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
 *
 * @file
 */
#pragma once

#include <Core/cuStinger.hpp>        //cuStingerInit
#include <Core/cuStingerTypes.cuh>   //Vertex, Edge

namespace cu_stinger_alg {
using cu_stinger::cuStingerInit;
using cu_stinger::id_t;
using cu_stinger::off_t;
using cu_stinger::degree_t;
using cu_stinger::Vertex;
using cu_stinger::Edge;

class Queue {
public:
    explicit Queue(const cuStingerInit& custinger_init,
                   float allocation_factor = 2.0f) noexcept;
    ~Queue() noexcept;

    __host__ void insert(id_t vertex_id) noexcept;
    __host__ void insert(const id_t* vertex_array, int size) noexcept;

    __host__ int size() noexcept;

    template<typename Operator, typename... TArgs>
    __host__ void traverseAndFilter(TArgs... args) noexcept;
private:
    const  cuStingerInit& _custinger_init;
    degree_t* _d_work             { nullptr };
    id_t*     _d_queue1           { nullptr };
    int2*     _d_queue2           { nullptr };
    int*      _d_queue_counter    { nullptr };
    int       _num_queue_vertices { 0 };
    int       _num_queue_edges    { 0 };
};

//------------------------------------------------------------------------------

class Allocate {
public:
    template<typename T>
    Allocate(T*& pointer, size_t num_items) noexcept;
    ~Allocate() noexcept;
private:
    void* _pointer;
};

} // namespace cu_stinger_alg

#include "Queue.i.cuh"

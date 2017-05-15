/**
 * @brief cuStinger algorithms interface
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

#include "Core/LoadBalancing/BinarySearch.cuh"   //load_balacing::BinarySearch
#include <Core/cuStinger.hpp>                    //custinger::cuStinger

/**
 * @brief cuStinger algorithms namespace
 */
namespace custinger_alg {

/**
 * @brief Abstract class for cuStinger static algorithms
 * @remark the user must extend this class to be full compliant with the
 *         cuStinger interface
 */
class StaticAlgorithm {
public:
    /**
     * @brief Default costructor
     * @param[in] custinger_ cuStinger instance on which the algorithm is run
     * @remark the cuStinger instance reference can be used in also methods
     *          throws `custinger` field
     */
    explicit StaticAlgorithm(custinger::cuStinger& custinger_) noexcept;

    /**
     * @brief Decostructor
     * @remark the user should implements this method to ensure that all
     *         the resources are deallocated when the class instance is not
     *         used anymore
     */
    virtual ~StaticAlgorithm() noexcept = 0;

    /**
     * @brief cuStinger static algorithm implementation
     * @remark the user must implements the main part of the algorithm in this
     *         method
     */
    virtual void run() = 0;

    /**
     * @brief reset all algorithm-dependent variables
     * @remark the user must implements this method to reset the instaance of
     *         algorithm to the initial state. After this method call the
     *         the algorithm instance is ready for the next execution
     * @remark the method should be called in the costructor
     */
    virtual void reset() = 0;

    /**
     * @brief release all the resources associeted with the algorithm instance
     * @remark the user should implements this method in very similar way of the
     *         decostructor. The decostructor can also calls this method.
     *         After the method call, the class instance cannot be used other
     *         time
     * @warning all pointers should be set to `nullptr` to allow safe
     *          deallocation of the instance with the decostructor method
     */
    virtual void release() = 0;

    /**
     * @brief validation of the (GPU) algorithm results
     * @return `true` if the host result is equal to the device result, `false`
     *          otherwise
     * @remark the user must implements this method to validate all possible
     *         exectutions of the algorithm. This method should implements
     *         the sequential algorithm.
     */
    virtual bool validate() = 0;

    /**
     * @brief register the algorithm-dependent data to the instance
     * @tparam T type of the algorithm-dependent data (deduced)
     * @param[in] data host-side algorithm-dependent data
     * @return device pointer to the algorithm-dependent data
     * @remark the user must call this method in the costructor to enable
     *         the `syncDeviceWithHost()` and `syncHostWithDevice()` methods
     */
    template<typename T>
    T* register_data(T& data) noexcept;

    /**
     * @brief Synchronize the Device with the Host
     * @@details Copy the algorithm-dependent data from the device to the host in
     *         the reference passed to `register_data()`
     * @pre    the user must call `register_data()` before
     * @remark the user should call this method after the main algorithm
     *         iteration
     */
    virtual void syncDeviceWithHost() noexcept final;

    /**
     * @brief Synchronize the Host with the Device
     * @@details Copy the algorithm-dependent data from the host to the device in
     *         the poiter return by `register_data()`
     * @pre    the user must call `register_data()` before
     * @remark the user should call this method before the main algorithm
     *         iteration
     */
    virtual void syncHostWithDevice() noexcept final;

protected:
    /**
     * @brief Load balancing class
     * @details provide the load balancing fucntionalities to efficiently
     *          traverse the graph
     */
    load_balacing::BinarySearch load_balacing;

    /**
     * @brief custinger reference
     */
    custinger::cuStinger& custinger;

private:
    size_t _data_size     { 0 };
    void*  _h_ptr         { nullptr };
    void*  _d_ptr         { nullptr };
    bool   _is_registered { false };
};

//==============================================================================

class Allocate {
public:
    template<typename T>
    explicit Allocate(T*& pointer, size_t num_items) noexcept;

    ~Allocate() noexcept;
private:
    void* _pointer;
};

} // namespace custinger_alg

#include "cuStingerAlg.i.cuh"

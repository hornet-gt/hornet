/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/Timer.hpp"
#include <cuda_runtime.h>       //cudaEvent_t

namespace timer {

template<typename ChronoPrecision>
class Timer<DEVICE, ChronoPrecision> :
        public TimerBase<DEVICE, ChronoPrecision> {
public:
    using TimerBase<DEVICE, ChronoPrecision>::print;
    using TimerBase<DEVICE, ChronoPrecision>::duration;
    using TimerBase<DEVICE, ChronoPrecision>::total_duration;
    using TimerBase<DEVICE, ChronoPrecision>::average;
    using TimerBase<DEVICE, ChronoPrecision>::std_deviation;

    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT);
    ~Timer();
    virtual void start() final;
    virtual void stop()  final;
private:
    using TimerBase<DEVICE, ChronoPrecision>::_time_elapsed;
    using TimerBase<DEVICE, ChronoPrecision>::_time_squared;
    using TimerBase<DEVICE, ChronoPrecision>::_total_time_elapsed;
    using TimerBase<DEVICE, ChronoPrecision>::_num_executions;
    using TimerBase<DEVICE, ChronoPrecision>::_start_flag;

    cudaEvent_t _start_event, _stop_event;
};

} // namespace timer

#include "impl/Timer.i.cuh"

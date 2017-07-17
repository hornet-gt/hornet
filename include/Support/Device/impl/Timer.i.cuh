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
 * @date July, 2017
 * @version v1.3
 */
#include <cassert>

namespace timer {

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>
::Timer(int decimals, int space, xlib::Color color) noexcept :
                    TimerBase<DEVICE, ChronoPrecision>(decimals, space, color) {
    cudaEventCreate(&_start_event);
    cudaEventCreate(&_stop_event);
}

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>::~Timer() noexcept {
    cudaEventDestroy(_start_event);
    cudaEventDestroy(_stop_event);
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = false;
    cudaEventRecord(_start_event, 0);    
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::stop() noexcept {
    float cuda_time_elapsed;
    cudaEventRecord(_stop_event, 0);
    cudaEventSynchronize(_stop_event);
    cudaEventElapsedTime(&cuda_time_elapsed, _start_event, _stop_event);
    auto  time_ms = timer::milli(cuda_time_elapsed);
    _time_elapsed = std::chrono::duration_cast<ChronoPrecision>(time_ms);
    register_time();
}

} // namespace timer

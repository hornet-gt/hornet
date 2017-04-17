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
namespace timer {

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>
::Timer(int decimals, int space, xlib::Color color) :
                    TimerBase<DEVICE, ChronoPrecision>(decimals, space, color) {
    cudaEventCreate(&_start_event);
    cudaEventCreate(&_stop_event);
}

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>::~Timer() {
    cudaEventDestroy(_start_event);
    cudaEventDestroy(_stop_event);
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::start() {
    assert(!_start_flag);
    cudaEventRecord(_start_event, 0);
    assert(_start_flag = true);
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::stop() {
    assert(_start_flag);
    cudaEventRecord(_stop_event, 0);
    cudaEventSynchronize(_stop_event);
    float cuda_time_elapsed;
    cudaEventElapsedTime(&cuda_time_elapsed, _start_event, _stop_event);
    _time_elapsed  = ChronoPrecision(cuda_time_elapsed);
    _time_squared += _time_elapsed * _time_elapsed.count();
    _total_time_elapsed += _time_elapsed;
    _num_executions++;
    assert(_start_flag = false);
}

} // namespace timer

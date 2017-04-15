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

#include <chrono>           //std::chrono::duration
#include <string>           //std::string
#if defined(__linux__)
    #include <sys/times.h>  //::tms
#endif

#define COLOR

#if defined(COLOR)
    #include "Support/PrintExt.hpp"
#else
namespace xlib { struct Color { enum TMP { FG::FG_DEFAULT }; }; }
inline std::ostream& operator<<(std::ostream& os, Color mod) { return os; }
#endif

namespace timer {

/// @brief chrono precision : microseconds
using   micro = typename std::chrono::duration<float, std::micro>;
/// @brief default chrono precision (milliseconds)
using   milli = typename std::chrono::duration<float, std::milli>;
/// @brief chrono precision : seconds
using seconds = typename std::chrono::duration<float, std::ratio<1>>;
/// @brief chrono precision : minutes
using minutes = typename std::chrono::duration<float, std::ratio<60>>;
/// @brief chrono precision : minutes
using   hours = typename std::chrono::duration<float, std::ratio<3600>>;

/**
 * @brief timer types
 */
enum timer_type {  HOST = 0       /// Wall (real) clock host time
                 , CPU  = 1       /// CPU User time
            #if defined(__linux__)
                 , SYS  = 2       /// User/Kernel/System time
            #endif
                 , DEVICE = 3     /// GPU device time
};

/**
 * @brief Timer class
 * @tparam type Timer type (default = HOST)
 * @tparam ChronoPrecision time precision
 */
template<timer_type type, typename ChronoPrecision>
class TimerBase {
    template<typename>
    struct is_duration : std::false_type {};

    template<typename T, typename R>
    struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

    static_assert(is_duration<ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
protected:
    ChronoPrecision _time_elapsed       {};
    ChronoPrecision _time_squared       {};
    ChronoPrecision _total_time_elapsed {};
    const int _decimals;
    const int _space;
    int       _num_executions { 0 };
    const xlib::Color _default_color;
    bool      _start_flag           { false };

    /**
     * @brief Default costructor
     * @param[in] decimals precision to print the time elapsed
     * @param[in] space space for the left alignment
     * @param[in] color color of print
     */
    explicit TimerBase(int decimals = 1, int space = 15,
                       xlib::Color color = xlib::Color::FG_DEFAULT);
    virtual ~TimerBase() = default;

    /**
     * @brief Start the timer
     */
    virtual void start() = 0;

    /**
     * @brief Stop the timer
     */
    virtual void stop() = 0;

    /**
     * @brief Get the time elapsed between start() and stop() calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float duration() const final;

    /**
     * @brief Get the time elapsed between the first start() and the last stop()
     *        calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float total_duration() const final;

    /**
     * @brief Get the average time elapsed between the first start() and the
     *        last stop() calls
     * @return average duration
     */
    virtual float average() const final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float std_deviation() const final;

    /**
     * @brief Print the time elapsed between start() and stop() calls
     * @param[in] str print string \p str before the time elapsed
     * @warning if start() and stop() not invoked undefined behavior
     */
    virtual void print(const std::string& str = "") const;              //NOLINT
};

template<timer_type type, typename ChronoPrecision = milli>
class Timer;

template<typename ChronoPrecision>
class Timer<HOST, ChronoPrecision> final :
            public TimerBase<HOST, ChronoPrecision> {
public:
    using TimerBase<HOST, ChronoPrecision>::print;
    using TimerBase<HOST, ChronoPrecision>::duration;
    using TimerBase<HOST, ChronoPrecision>::total_duration;
    using TimerBase<HOST, ChronoPrecision>::average;
    using TimerBase<HOST, ChronoPrecision>::std_deviation;

    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT);
    virtual void start() final;
    virtual void stop()  final;
private:
    using TimerBase<HOST, ChronoPrecision>::_time_elapsed;
    using TimerBase<HOST, ChronoPrecision>::_time_squared;
    using TimerBase<HOST, ChronoPrecision>::_total_time_elapsed;
    using TimerBase<HOST, ChronoPrecision>::_num_executions;
    using TimerBase<HOST, ChronoPrecision>::_start_flag;

    std::chrono::system_clock::time_point  _start_time         {};
    std::chrono::system_clock::time_point  _stop_time          {};
};

template<typename ChronoPrecision>
class Timer<CPU, ChronoPrecision> final :
            public TimerBase<CPU, ChronoPrecision> {
public:
    using TimerBase<CPU, ChronoPrecision>::print;
    using TimerBase<CPU, ChronoPrecision>::duration;
    using TimerBase<CPU, ChronoPrecision>::total_duration;
    using TimerBase<CPU, ChronoPrecision>::average;
    using TimerBase<HOST, ChronoPrecision>::std_deviation;

    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT);
    virtual void start() final;
    virtual void stop()  final;
private:
    using TimerBase<CPU, ChronoPrecision>::_time_elapsed;
    using TimerBase<HOST, ChronoPrecision>::_time_squared;
    using TimerBase<CPU, ChronoPrecision>::_total_time_elapsed;
    using TimerBase<CPU, ChronoPrecision>::_num_executions;
    using TimerBase<CPU, ChronoPrecision>::_start_time;
    using TimerBase<CPU, ChronoPrecision>::_stop_time;
    using TimerBase<CPU, ChronoPrecision>::_start_flag;

    std::clock_t _start_clock { 0 };
    std::clock_t _stop_clock  { 0 };
};

#if defined(__linux__)

template<typename ChronoPrecision>
class Timer<SYS, ChronoPrecision> final :
            public TimerBase<SYS, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT);
    virtual void start() final;
    virtual void stop()  final;
    virtual void print(const std::string& str = "") const final;        //NOLINT
private:
    using TimerBase<SYS, ChronoPrecision>::_start_time;
    using TimerBase<SYS, ChronoPrecision>::_stop_time;
    using TimerBase<SYS, ChronoPrecision>::_start_flag;
    using TimerBase<SYS, ChronoPrecision>::_default_color;
    using TimerBase<SYS, ChronoPrecision>::_space;
    using TimerBase<SYS, ChronoPrecision>::_decimals;

    struct ::tms _start_TMS {};
    struct ::tms _end_TMS   {};
};

#endif

} // namespace timer

#include "impl/Timer.i.hpp"

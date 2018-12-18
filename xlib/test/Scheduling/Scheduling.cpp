/*------------------------------------------------------------------------------
Copyright © 2016 by Federico Busato

OSScheduling is provided under the terms of The MIT License (MIT):

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
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <map>
#include <cassert>

using timev_t = int;
using  proc_t = int;

using         TimeInfo = std::pair<timev_t, timev_t>;
using     scheduling_t = std::vector<std::vector<TimeInfo>>;
using             MapT = std::pair<timev_t, proc_t>;
using         TimePair = std::pair<timev_t, timev_t>;
using            MapT2 = std::pair<TimePair, proc_t>;
using MinPriorityQueue = std::map<TimePair, proc_t>;

namespace {

void noPreemptive(const std::vector<timev_t>& arrival_time,
                  const std::vector<timev_t>& cpu_burst,
                  const std::vector<timev_t>& to_minimize,
                  scheduling_t&               scheduling) {

    const int n_of_processes = arrival_time.size();
    std::multimap<timev_t, proc_t> queue;
    for (int i = 0; i < n_of_processes; i++)
        queue.insert(MapT(to_minimize[i], i));

    timev_t time = 0;
    while (!queue.empty()) {
        auto item = queue.begin();
        queue.erase(item);
        proc_t process = (*item).second;

        time = std::max(time, arrival_time[process]);
        auto time_info = TimeInfo(time, time + cpu_burst[process]);
        scheduling[process].push_back(time_info);
        time += cpu_burst[process];
    }
}

} // namespace anonymous

void FCFS(const std::vector<timev_t>& arrival_time,
          const std::vector<timev_t>& cpu_burst,
          scheduling_t&               scheduling) {

    noPreemptive(arrival_time, cpu_burst, arrival_time, scheduling);
}

void SJF(const std::vector<timev_t>& arrival_time,
         const std::vector<timev_t>& cpu_burst,
         scheduling_t&               scheduling) {

    noPreemptive(arrival_time, cpu_burst, cpu_burst, scheduling);
}

namespace {

template<typename iterator_t>
const iterator_t find_min(const iterator_t& start,
                          const iterator_t& end,
                          const std::vector<timev_t>& cpu_burst,
                          timev_t time,
                          timev_t min_value
                            = std::numeric_limits<timev_t>::max()) {

    iterator_t selected = end;
    for (auto it = start; (*it).first <= time && it != end; it++) {
        timev_t arrival_tmp = cpu_burst[(*it).second];
        if (arrival_tmp < min_value) {
            min_value = arrival_tmp;
            selected = it;
        }
    }
    return selected;
}

} // namespace anonymous

void SRTF(const std::vector<int>& arrival_time,
          const std::vector<int>& cpu_burst,
          scheduling_t&           scheduling) {

    const int n_of_processes = arrival_time.size();
    std::multimap<timev_t, proc_t> preemption_queue;

    std::multimap<timev_t, proc_t> queue;
    for (int i = 0; i < n_of_processes; i++)
        queue.insert(MapT(arrival_time[i], i));

    timev_t time = 0;
    while (!queue.empty() || !preemption_queue.empty()) {
        auto it = find_min(queue.begin(), queue.end(), cpu_burst, time);

        timev_t burst;
        std::multimap<timev_t, proc_t>::iterator selected_it;
        if (it == queue.end() && !preemption_queue.empty()) {
            selected_it = preemption_queue.begin();
            burst = (*selected_it).first;
            preemption_queue.erase(selected_it);
        } else {
            selected_it = it == queue.end() ? queue.begin() : it;
            burst = cpu_burst[(*selected_it).second];
            queue.erase(selected_it);
        }
        proc_t selected = (*selected_it).second;

        time = std::max(time, arrival_time[selected]);
        timev_t requested = time + burst;

        //std::cout << "selected " << selected
        //          << " requested " << requested << std::endl;

        //arrival_time < requested
        const auto it_next = find_min(queue.begin(), queue.end(), cpu_burst,
                                      requested, burst);

        time_t cpu_burst_real;
        if (it_next != queue.end()) {
            proc_t    next = (*it_next).second;
            timev_t remain = requested - arrival_time[next];
            cpu_burst_real = burst - remain;
            //std::cout << "\tnext " << next << " requested " << requested
            //          << " -> reinsert " << remain << std::endl;
            preemption_queue.insert(MapT(remain, selected));
        } else
            cpu_burst_real = burst;

        auto time_info = TimeInfo(time, time + cpu_burst_real);
        scheduling[selected].push_back(time_info);
        time += cpu_burst_real;
    }
}

void printScheduling(const std::vector<std::string>& processes,
                     const std::vector<int>&         arrival_time,
                     const std::vector<int>&         cpu_burst,
                     scheduling_t&                   scheduling) {

    const int n_of_processes = processes.size();

    timev_t max_time = -1;
    for (int i = 0; i < n_of_processes; i++) {
        assert(scheduling[i].size() > 0);
        max_time = std::max(max_time, (*std::prev(scheduling[i].end())).second);
    }

    std::cout << "\n      ";
    for (int i = 0; i <= max_time; i++)
        std::cout << std::left << std::setw(3) << i;
    std::cout << "\n";

    std::cout << "      ";
    for (int i = 0; i < max_time; i++)
        std::cout << "|--";
    std::cout << "|\n";

    //--------------------------------------------------------------------------

    for (int i = 0; i < n_of_processes; i++) {
        std::cout << std::setw(6) << std::left << processes[i] << "|";
        int init = 0;
        for (const auto& it : scheduling[i]) {
            //std::cout << it.first << " " << it.second << std::endl;
            for (int j = init; j < it.first; j++)
                std::cout << "   ";
            for (int j = it.first; j < it.second; j++)
                std::cout << "###";
            init = it.second;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    scheduling.clear();
    scheduling.resize(n_of_processes);
}

int main(int argc, char* argv[]) {
    std::ifstream fin(argv[1]);

    std::vector<std::string> processes;
    std::vector<int>         arrival_time;
    std::vector<int>         cpu_burst;

    while (fin.good()) {
        std::string name;
        int arrival_time_value, cpu_burst_value;
        fin >> name >> arrival_time_value >> cpu_burst_value;
        processes.push_back(name);
        arrival_time.push_back(arrival_time_value);
        cpu_burst.push_back(cpu_burst_value);
    }
    fin.close();

    const int n_of_processes = processes.size();
    scheduling_t scheduling(n_of_processes);

    //--------------------------------------------------------------------------
    /*FCFS(arrival_time, cpu_burst, scheduling);

    printScheduling(processes, arrival_time, cpu_burst, scheduling);

    //--------------------------------------------------------------------------
    SJF(arrival_time, cpu_burst, scheduling);

    printScheduling(processes, arrival_time, cpu_burst, scheduling);*/
    //--------------------------------------------------------------------------
    SRTF(arrival_time, cpu_burst, scheduling);

    printScheduling(processes, arrival_time, cpu_burst, scheduling);
}

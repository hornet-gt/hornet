#pragma once

#include "cuStingerAlg.hpp"
#include <cuStinger.hpp>

namespace custinger_alg {

using dist_t = int;
using custinger::vid_t;
using custinger::eoff_t;

struct BfsData {
    BfsData() : queue(10) {}

	TwoLevelQueue<vid_t> queue;
    dist_t* distances;
	dist_t  current_level;
};

class BfsTopDown final : public StaticAlgorithm {
public:
    explicit BfsTopDown(const custinger::cuStinger& custinger);
    ~BfsTopDown();

    void set_parameters(vid_t bfs_source);
	void reset()    override;
	void run()      override;
	void release()  override;
    bool validate() override;
private:
    vid_t   bfs_source;
	BfsData bfs_data;
};

} // namespace custinger_alg

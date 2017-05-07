#pragma once

#include "cuStingerAlg.hpp"

namespace custinger_alg {

using dist_t = int;

struct BfsData {
    BfsData(cuStinger& custinger) : queue(custinger) {}

	TwoLevelQueue<vid_t> queue;
    dist_t* distances;
	dist_t  current_level;
};

class BfsTopDown final : public StaticAlgorithm {
public:
    explicit BfsTopDown(cuStinger& custinger);
    ~BfsTopDown();

    void set_parameters(vid_t source);
	void reset()    override;
	void run()      override;
	void release()  override;
    bool validate() override;
private:
	BfsData  host_bfs_data;
	BfsData* device_bfs_data { nullptr };
    vid_t    bfs_source      { 0 };
};

} // namespace custinger_alg

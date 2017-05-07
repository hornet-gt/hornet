#pragma once

#include "cuStingerAlg.hpp"

namespace custinger_alg {

using dist_t = int;

class BfsTopDown2 final : public StaticAlgorithm {
public:
    explicit BfsTopDown2(cuStinger& custinger);
    ~BfsTopDown2();

    void set_parameters(vid_t source);
	void reset()    override;
	void run()      override;
	void release()  override;
    bool validate() override;
private:
    TwoLevelQueue<vid_t> queue;
    dist_t* d_distances   { nullptr };
    vid_t   bfs_source    { 0 };
    dist_t  current_level { 0 };
};

} // namespace custinger_alg

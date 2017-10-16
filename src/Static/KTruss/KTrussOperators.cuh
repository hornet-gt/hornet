#pragma once

namespace hornets_nest {

struct Init {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vid_t           src = vertex.id();
        kt().is_active[src] = 1;
    }
};

//------------------------------------------------------------------------------

struct FindUnderK {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        if (kt().is_active[src] == 0)
            return;
        if (vertex.degree() == 0) {
            kt().is_active[src] = 0;
            return;
        }
        for (vid_t adj = 0; adj < vertex.degree(); adj++) {
            auto edge = vertex.edge(adj);
            int   pos = kt().offset_array[src] + adj;
            if (kt().triangles_per_edge[pos] < kt().max_K - 2) {
                int       spot = atomicAdd(&(kt().counter), 1);
                kt().src[spot] = src;
                kt().dst[spot] = edge.dst_id();
            }
        }
    }
};

//------------------------------------------------------------------------------

struct FindUnderKDynamic {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        if(kt().is_active[src] == 0)
            return;
        if(vertex.degree() == 0) {
            kt().is_active[src] = 0;
            return;
        }
        for (vid_t adj = 0; adj < vertex.degree(); adj++) {
            auto edge = vertex.edge(adj);
            if (edge.weight() < kt().max_K - 2) {
                int       spot = atomicAdd(&(kt().counter), 1);
                kt().src[spot] = src;
                kt().dst[spot] = edge.dst_id();
            }
        }
    }
};

//------------------------------------------------------------------------------

struct QueueActive {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        if (vertex.degree() == 0 && !kt().is_active[src])
            kt().is_active[src] = 0;
        else
            kt().active_queue.insert(src);
    }
};

//------------------------------------------------------------------------------

struct CountActive {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        if (vertex.degree() == 0 && !kt().is_active[src])
            kt().is_active[src] = 0;
        else
            atomicAdd(&(kt().active_vertices), 1);
    }
};

//------------------------------------------------------------------------------

struct ResetWeights {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        int pos = kt().offset_array[vertex.id()];

        for (vid_t adj = 0; adj < vertex.degree(); adj++)
            vertex.edge(adj).set_weight(kt().triangles_per_edge[pos + adj]); //!!!
    }
};

} // namespace hornets_nest

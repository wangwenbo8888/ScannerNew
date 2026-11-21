#pragma once
// edge_table.hpp - [S1'] shared topology table for SR (speedopt R9B).
// One build, three consumers: orient BFS (read-only) -> ownership handed to
// edge_flip (mutable) -> NOT for stats (E6 kill order, see R3A).
// Bitwise replica of the legacy edge_flip.cpp builder: reserve nt*3 (both
// containers), NO self-skip, operator[] first-touch insert in (t asc, k 0..2)
// order, t2 fill, nmExtra++. The edges-vector order IS the flip pass scan
// order - load-bearing, do not reorder.
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sr {


struct EdgeR {
    uint32_t a, b;
    int32_t t1, t2;
};

inline uint64_t ekeyT(uint32_t a, uint32_t b) {
    return a < b ? ((uint64_t)a << 32) | b : ((uint64_t)b << 32) | a;
}

struct EdgeTable {
    std::unordered_map<uint64_t, uint32_t> eidx;
    std::vector<EdgeR> edges;                  // first-touch order = flip scan order
    std::unordered_map<uint32_t, int> nmExtra; // 3rd+ face count per edge idx
};

inline EdgeTable buildEdgeTable(const uint32_t* T, size_t nt) {
    EdgeTable tb;
    tb.eidx.reserve(nt * 3);
    tb.edges.reserve(nt * 3);
    for (size_t t = 0; t < nt; t++) {
        for (int k = 0; k < 3; k++) {
            uint32_t a = T[t * 3 + k], b = T[t * 3 + (k + 1) % 3];
            uint64_t k2 = ekeyT(a, b);
            auto it = tb.eidx.find(k2);
            if (it == tb.eidx.end()) {
                tb.eidx[k2] = (uint32_t)tb.edges.size();
                tb.edges.push_back({a, b, (int32_t)t, -1});
            } else if (tb.edges[it->second].t2 < 0) {
                tb.edges[it->second].t2 = (int32_t)t;
            } else {
                tb.nmExtra[it->second]++;
            }
        }
    }
    return tb;
}

} // namespace sr

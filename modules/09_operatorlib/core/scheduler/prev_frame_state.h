#pragma once
#include <atomic>
#include <memory>
#include <vector>
#include <cstdint>

namespace calib {

struct MarkerPoint3D {
    double x, y, z;
    double nx, ny, nz;
    int globalId;
};

struct AtomicFrameState {
    std::vector<MarkerPoint3D> rawPoints;
    std::vector<double> normals;
    std::vector<int> globalIds;

    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double T[3] = {0,0,0};
    uint64_t frameId = 0;

    static std::shared_ptr<AtomicFrameState> load(
        const std::shared_ptr<AtomicFrameState>& src) {
        return std::atomic_load(&src);
    }

    static void store(std::shared_ptr<AtomicFrameState>& dst,
                      std::shared_ptr<AtomicFrameState> state) {
        std::atomic_store(&dst, std::move(state));
    }

    bool isFirstFrame() const { return frameId == 0; }
};

} // namespace calib

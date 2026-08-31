/**
 * @file marker_match_cpu.cpp
 * @brief 标记点匹配算子 - 实现文件
 *
 * 算法流程：
 *   1. 输入左右相机立体矫正后的亚像素标记点集
 *   2. 按 Y 坐标排序建立 SoA 布局（缓存友好）
 *   3. 自适应容差计算（基于点集密度）
 *   4. 滑动窗口极线约束匹配（唯一性检测）
 *   5. 置信度评估和统计信息收集
 *   6. 输出视差和匹配置信度
 */

#include "marker_match_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cassert>
#include <numeric>
#include <stdexcept>

using namespace calib;

#ifdef _OPENMP
#include <omp.h>

#endif

namespace calib {
OperatorInfo getMarkerMatchCPUInfo() {
    return OperatorInfo{"MarkerMatchCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}
}

CALIB_DEFINE_LOG_TAG(08, MarkerMatchCPU);

namespace {

constexpr float EPSILON = 1e-6f;
constexpr float BUCKET_SIZE_MIN = 1e-6f;
constexpr int MAX_BUCKETS = 1000;
constexpr float Y_CONFIDENCE_WEIGHT = 0.7f;
constexpr float CONSISTENCY_WEIGHT = 0.3f;
constexpr float MAX_DEVIATION_TOLERANCE_MULTIPLIER = 50.0f;

#pragma pack(push, 1)
struct IndexPoint {
    float y;
    float x;
    uint32_t original_idx;

    IndexPoint() : y(0.0f), x(0.0f), original_idx(0) {}

    explicit IndexPoint(size_t idx, const cv::Point2f& pt)
        : y(pt.y), x(pt.x), original_idx(static_cast<uint32_t>(idx)) {}

    bool operator<(const IndexPoint& other) const {
        return y < other.y;
    }
};
#pragma pack(pop)

struct SortedPointsArray {
    std::vector<float> y;
    std::vector<float> x;
    std::vector<uint32_t> idx;

    SortedPointsArray() = default;

    explicit SortedPointsArray(size_t capacity) {
        y.reserve(capacity);
        x.reserve(capacity);
        idx.reserve(capacity);
    }

    void clear() noexcept {
        y.clear();
        x.clear();
        idx.clear();
    }

    void reserve(size_t capacity) {
        y.reserve(capacity);
        x.reserve(capacity);
        idx.reserve(capacity);
    }

    size_t size() const noexcept { return y.size(); }
    bool empty() const noexcept { return y.empty(); }

    void push_back(float y_val, float x_val, uint32_t index) {
        y.push_back(y_val);
        x.push_back(x_val);
        idx.push_back(index);
    }

    void fromPointsSorted(const std::vector<cv::Point2f>& points) {
        clear();
        size_t n = points.size();
        reserve(n);

        std::vector<IndexPoint> temp(n);
        for (size_t i = 0; i < n; ++i) {
            temp[i] = IndexPoint(i, points[i]);
        }
        std::sort(temp.begin(), temp.end());

        for (const auto& pt : temp) {
            push_back(pt.y, pt.x, pt.original_idx);
        }
    }

    const float* yData() const noexcept { return y.data(); }
};

struct YZoneIndex {
    std::vector<size_t> bucket_starts;
    std::vector<size_t> bucket_ends;
    float min_y;
    float bucket_size;
    int num_buckets;

    YZoneIndex() : min_y(0), bucket_size(1.0f), num_buckets(0) {}

    inline int getBucketIndex(float y_coord) const {
        if (num_buckets <= 0) return -1;
        if (std::isnan(y_coord)) return -1;
        int bidx = static_cast<int>((y_coord - min_y) / bucket_size);
        if (bidx < 0) return 0;
        if (bidx >= num_buckets) return num_buckets - 1;
        return bidx;
    }

    inline bool isValid() const { return num_buckets > 0 && !bucket_starts.empty(); }
};

size_t findLowerBound(const float* y_coords, size_t count, float target) {
    if (count == 0) return 0;
    if (count > 16) {
        size_t lo = 0, hi = count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (y_coords[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
    for (size_t i = 0; i < count; ++i) {
        if (y_coords[i] >= target) return i;
    }
    return count;
}

void buildYZoneIndex(const std::vector<IndexPoint>& sorted_points,
                     YZoneIndex& index,
                     float bucket_size) {
    if (sorted_points.empty()) {
        index.num_buckets = 0;
        return;
    }

    if (bucket_size <= BUCKET_SIZE_MIN) {
        bucket_size = BUCKET_SIZE_MIN;
    }

    index.min_y = sorted_points.front().y;
    float max_y = sorted_points.back().y;
    index.bucket_size = bucket_size;

    float y_range = max_y - index.min_y;
    index.num_buckets = static_cast<int>(y_range / bucket_size) + 1;

    if (index.num_buckets > MAX_BUCKETS) {
        index.num_buckets = MAX_BUCKETS;
        index.bucket_size = (y_range > EPSILON) ? (y_range / static_cast<float>(MAX_BUCKETS - 1)) : 1.0f;
    }

    index.bucket_starts.resize(index.num_buckets);
    index.bucket_ends.resize(index.num_buckets);

    size_t current_idx = 0;
    for (int b = 0; b < index.num_buckets; ++b) {
        float bucket_low = index.min_y + b * index.bucket_size;
        float bucket_high = bucket_low + index.bucket_size;

        while (current_idx < sorted_points.size() && sorted_points[current_idx].y < bucket_low) {
            current_idx++;
        }
        index.bucket_starts[b] = current_idx;

        while (current_idx < sorted_points.size() && sorted_points[current_idx].y < bucket_high) {
            current_idx++;
        }
        index.bucket_ends[b] = current_idx;
    }
}

float calculateAdaptiveTolerance(const std::vector<cv::Point2f>& points,
                                 float base_tolerance,
                                 float density_high,
                                 float density_low,
                                 float dense_scale,
                                 float sparse_scale) {
    if (points.empty()) return base_tolerance;

    float min_y = points[0].y;
    float max_y = points[0].y;
    for (const auto& pt : points) {
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }

    float y_range = max_y - min_y;
    if (y_range < EPSILON) return base_tolerance;

    float density = static_cast<float>(points.size()) / y_range;

    if (density > density_high) {
        return base_tolerance * dense_scale;
    } else if (density < density_low) {
        return base_tolerance * sparse_scale;
    }

    return base_tolerance;
}

float sumArray(const float* arr, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += arr[i];
    }
    return sum;
}

float calculateDisparityConsistency(float disparity,
                                    const float* neighbor_disparities,
                                    size_t neighbor_count,
                                    float y_tolerance) {
    if (neighbor_count == 0 || neighbor_disparities == nullptr) {
        return 1.0f;
    }

    float avg_neighbor = sumArray(neighbor_disparities, neighbor_count) /
                         static_cast<float>(neighbor_count);

    float deviation = std::abs(disparity - avg_neighbor);
    float max_deviation = y_tolerance * MAX_DEVIATION_TOLERANCE_MULTIPLIER;

    if (max_deviation < EPSILON) {
        return (deviation < EPSILON) ? 1.0f : 0.0f;
    }

    if (deviation >= max_deviation) {
        return 0.0f;
    }

    float ratio = deviation / max_deviation;
    return 1.0f - ratio * ratio;
}

float calculateConfidence(float center_y, float matched_y, float tolerance,
                          float disparity,
                          const float* neighbor_disparities,
                          size_t neighbor_count,
                          float y_tolerance) {
    if (tolerance <= 0.0f) return 0.0f;

    float y_distance = std::abs(center_y - matched_y);
    if (y_distance > tolerance) return 0.0f;
    float y_confidence = 1.0f - (y_distance / tolerance);

    if (neighbor_count > 0 && neighbor_disparities != nullptr) {
        float consistency = calculateDisparityConsistency(
            disparity, neighbor_disparities, neighbor_count, y_tolerance);
        return y_confidence * Y_CONFIDENCE_WEIGHT + consistency * CONSISTENCY_WEIGHT;
    }

    return y_confidence;
}

void collectMatchStatistics(MarkerMatchStats& stats,
                            size_t matched_count,
                            size_t ambiguous_count,
                            size_t unvisited_count,
                            const std::vector<float>& disparities,
                            const std::vector<uint8_t>& valid_flags,
                            const std::vector<float>& confidence) {
    stats.matched_points = matched_count;
    stats.ambiguous_points = ambiguous_count;
    stats.unvisited_points = unvisited_count;
    stats.total_points = disparities.size();

    if (stats.total_points > 0) {
        stats.match_rate = static_cast<float>(stats.matched_points) /
                           static_cast<float>(stats.total_points);
    } else {
        stats.match_rate = 0.0f;
    }

    std::vector<float> valid_disps;
    valid_disps.reserve(stats.matched_points);
    float confidence_sum = 0.0f;

    for (size_t i = 0; i < disparities.size(); ++i) {
        if (valid_flags[i]) {
            valid_disps.push_back(disparities[i]);
            confidence_sum += confidence[i];
        }
    }

    stats.avg_disparity = 0.0f;
    stats.disparity_std = 0.0f;
    stats.avg_confidence = 0.0f;

    if (!valid_disps.empty()) {
        float sum = sumArray(valid_disps.data(), valid_disps.size());
        stats.avg_disparity = sum / static_cast<float>(valid_disps.size());
        stats.avg_confidence = confidence_sum / static_cast<float>(valid_disps.size());

        if (valid_disps.size() > 1) {
            float sq_sum = 0.0f;
            for (float d : valid_disps) {
                sq_sum += (d - stats.avg_disparity) * (d - stats.avg_disparity);
            }
            stats.disparity_std = std::sqrt(sq_sum / static_cast<float>(valid_disps.size() - 1));
        }

        if (stats.disparity_std < EPSILON) {
            stats.disparity_consistency = 1.0f;
        } else {
            float normalized_std = stats.disparity_std / (std::abs(stats.avg_disparity) + EPSILON);
            stats.disparity_consistency = std::max(0.0f, std::min(1.0f, 1.0f - normalized_std));
        }
    }
}

} // anonymous namespace

// ============================================================
// MarkerMatchCPUParams::validate
// ============================================================
void MarkerMatchCPUParams::validate() const {
    // 上限 1.0→10.0（2026-08-31 真机定版）：y_tolerance 实际口径为像素
    // （Execute 内 |center_y - matched_y| 直接像素比较），原 (0,1] 按归一化
    // 设计与用法矛盾——扫描链现场装配/温漂下真实标志点左右中心 y 差
    // 1~3px，上限 1 令匹配不稳定（真机 13 标志点帧 0~3 对波动）
    if (y_tolerance <= 0.0f || y_tolerance > 10.0f)
        throw std::invalid_argument("y_tolerance must be in (0, 10], got " + std::to_string(y_tolerance));
    if (num_threads < 0)
        throw std::invalid_argument("num_threads must be >= 0, got " + std::to_string(num_threads));
    if (prealloc_buffer_size == 0)
        throw std::invalid_argument("prealloc_buffer_size must be > 0");
    if (parallel_threshold == 0)
        throw std::invalid_argument("parallel_threshold must be > 0");
    if (max_points == 0)
        throw std::invalid_argument("max_points must be > 0");
    if (max_buffer_size == 0)
        throw std::invalid_argument("max_buffer_size must be > 0");
    if (max_buffer_size < max_points)
        throw std::invalid_argument("max_buffer_size must be >= max_points");
}

// ============================================================
// Impl
// ============================================================
struct MarkerMatchCPU::Impl {
    MarkerMatchCPUParams params_;
    MarkerMatchStats stats_;

    std::vector<size_t> left_window_buffer_;
    std::vector<size_t> right_window_buffer_;
    std::vector<float> valid_disparities_buffer_;
    std::vector<float> neighbor_disparities_buffer_;

    std::vector<IndexPoint> precomputed_right_sorted_;
    std::vector<cv::Point2f> precomputed_right_points_;
    SortedPointsArray precomputed_right_soa_;
    YZoneIndex right_yzone_index_;
    bool has_precomputed_ = false;

    SortedPointsArray left_soa_;
    SortedPointsArray right_soa_;

    std::vector<uint32_t> range_indices_buffer_;

#ifdef _OPENMP
    std::vector<std::vector<size_t>> thread_left_bufs_;
    std::vector<std::vector<size_t>> thread_right_bufs_;
    std::vector<std::vector<uint8_t>> thread_left_visited_;
    std::vector<std::vector<uint8_t>> thread_right_visited_;
    std::vector<std::vector<float>> thread_neighbor_disparities_;
    int last_num_threads_ = 0;
#endif

    bool warmed_up_ = false;
    int warmup_maxPoints_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const MarkerMatchCPUParams& params)
        : params_(params)
    {
        params_.validate();
        resizeBuffers(params_.prealloc_buffer_size);

#ifdef _OPENMP
        if (params_.num_threads > 0) {
            omp_set_num_threads(params_.num_threads);
        }
#endif
    }

    void resizeBuffers(size_t size) {
        if (size == 0 || size > params_.max_buffer_size)
            throw std::invalid_argument("Invalid buffer size");

        left_window_buffer_.resize(size);
        right_window_buffer_.resize(size);
        valid_disparities_buffer_.reserve(size);
        neighbor_disparities_buffer_.reserve(size);
    }

    void validatePoints(const std::vector<cv::Point2f>& points,
                        const std::string& side_name) const {
        for (size_t i = 0; i < points.size(); ++i) {
            if (std::isnan(points[i].x) || std::isnan(points[i].y))
                throw std::invalid_argument("NaN in " + side_name + " points at index " + std::to_string(i));
            if (std::isinf(points[i].x) || std::isinf(points[i].y))
                throw std::invalid_argument("Infinity in " + side_name + " points at index " + std::to_string(i));
        }
    }

    void validateInput(const std::vector<cv::Point2f>& left_points,
                       const std::vector<cv::Point2f>& right_points) const {
        if (left_points.empty() || right_points.empty())
            throw std::invalid_argument("Input point set is empty");
        if (left_points.size() > params_.max_points || right_points.size() > params_.max_points)
            throw std::invalid_argument("Point count exceeds max_points (" + std::to_string(params_.max_points) + ")");
        validatePoints(left_points, "left");
        validatePoints(right_points, "right");
    }

    MarkerMatchCPUResult ExecuteImpl(const std::vector<cv::Point2f>& left_points,
                   const std::vector<cv::Point2f>& right_points) {
        MarkerMatchCPUResult result;
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        auto total_start = std::chrono::high_resolution_clock::now();

        validateInput(left_points, right_points);

        stats_ = MarkerMatchStats();
        stats_.used_precomputed = false;

        result.disparities.assign(left_points.size(), std::numeric_limits<float>::quiet_NaN());
        result.valid_flags.assign(left_points.size(), static_cast<uint8_t>(0));
        result.confidence.assign(left_points.size(), 0.0f);
        result.centerMatches.assign(left_points.size(), -1);

        auto sort_start = std::chrono::high_resolution_clock::now();

        left_soa_.fromPointsSorted(left_points);
        right_soa_.fromPointsSorted(right_points);

        auto sort_end = std::chrono::high_resolution_clock::now();
        stats_.sort_time_ms = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();

        float y_tol = calculateAdaptiveTolerance(
            left_points, params_.y_tolerance,
            params_.density_threshold_high, params_.density_threshold_low,
            params_.dense_tolerance_scale, params_.sparse_tolerance_scale);

        size_t required_buffer_size = std::max(left_points.size(), right_points.size());
        if (left_window_buffer_.size() < required_buffer_size) {
            resizeBuffers(required_buffer_size);
        }
        if (range_indices_buffer_.size() < required_buffer_size) {
            range_indices_buffer_.resize(required_buffer_size);
        }

        auto match_start = std::chrono::high_resolution_clock::now();

        size_t matched_count = 0;
        size_t ambiguous_count = 0;
        size_t unvisited_count = 0;

        std::vector<uint8_t> left_visited(left_points.size(), 0);
        std::vector<uint8_t> right_visited(right_points.size(), 0);

        neighbor_disparities_buffer_.clear();
        neighbor_disparities_buffer_.reserve(left_points.size());

        const float* left_y = left_soa_.yData();
        const float* left_x = left_soa_.x.data();
        const uint32_t* left_idx = left_soa_.idx.data();

        const float* right_y = right_soa_.yData();
        const uint32_t* right_idx = right_soa_.idx.data();

        size_t left_size = left_soa_.size();
        size_t right_size = right_soa_.size();

        for (size_t i = 0; i < left_size; ++i) {
            uint32_t current_orig_idx = left_idx[i];

            if (left_visited[current_orig_idx]) {
                continue;
            }

            float center_y = left_y[i];
            float center_x = left_x[i];
            float low_y = center_y - y_tol;
            float high_y = center_y + y_tol;

            size_t left_count = 0;
            left_window_buffer_[left_count++] = current_orig_idx;

            for (size_t k = i + 1; k < left_size; ++k) {
                if (left_y[k] > high_y) break;
                uint32_t orig_idx = left_idx[k];
                if (!left_visited[orig_idx]) {
                    if (left_count >= left_window_buffer_.size())
                        throw std::runtime_error("Left window buffer overflow");
                    left_window_buffer_[left_count++] = orig_idx;
                }
            }

            size_t right_count = 0;
            size_t start_idx = findLowerBound(right_y, right_size, low_y);

            for (size_t j = start_idx; j < right_size; ++j) {
                if (right_y[j] > high_y) break;
                uint32_t orig_idx = right_idx[j];
                if (!right_visited[orig_idx]) {
                    if (right_count >= right_window_buffer_.size())
                        throw std::runtime_error("Right window buffer overflow");
                    right_window_buffer_[right_count++] = orig_idx;
                }
            }

            size_t N_L = left_count;
            size_t N_R = right_count;

            for (size_t j = 0; j < left_count; ++j) {
                left_visited[left_window_buffer_[j]] = 1;
            }
            for (size_t j = 0; j < right_count; ++j) {
                right_visited[right_window_buffer_[j]] = 1;
            }

            if (N_L == 1 && N_R == 1) {
                uint32_t right_orig_idx = static_cast<uint32_t>(right_window_buffer_[0]);
                float disparity = center_x - right_points[right_orig_idx].x;

                const float* neighbor_ptr = neighbor_disparities_buffer_.empty() ? nullptr :
                                            neighbor_disparities_buffer_.data();
                size_t neighbor_count = neighbor_disparities_buffer_.size();

                float conf = calculateConfidence(
                    center_y, right_points[right_orig_idx].y, y_tol,
                    disparity, neighbor_ptr, neighbor_count, params_.y_tolerance);

                result.disparities[current_orig_idx] = disparity;
                result.valid_flags[current_orig_idx] = 1;
                result.confidence[current_orig_idx] = conf;
                result.centerMatches[current_orig_idx] = static_cast<int>(right_orig_idx);

                neighbor_disparities_buffer_.push_back(disparity);
                matched_count++;
            } else if (N_L > 1 || N_R > 1) {
                ambiguous_count += N_L;
            }
        }

        for (size_t i = 0; i < left_points.size(); ++i) {
            if (!left_visited[i]) unvisited_count++;
        }

        auto match_end = std::chrono::high_resolution_clock::now();
        stats_.match_time_ms = std::chrono::duration<double, std::milli>(match_end - match_start).count();

        stats_.num_threads_used = 1;

        collectMatchStatistics(stats_, matched_count, ambiguous_count, unvisited_count,
                               result.disparities, result.valid_flags, result.confidence);

        auto total_end = std::chrono::high_resolution_clock::now();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        result.success = true;
        result.message = "Matching completed";
        result.qualityFlag = calib::QualityFlag::Normal;

        if (stats_.match_rate < 0.3f && stats_.total_points > 5) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Low match rate: " + std::to_string(static_cast<int>(stats_.match_rate * 100)) + "%";
        }

        return result;
    }

    void ExecuteWithReferenceImpl(const std::vector<cv::Point2f>& left_points,
                                MarkerMatchCPUResult& result) {
        auto total_start = std::chrono::high_resolution_clock::now();

        if (!has_precomputed_) {
            throw std::invalid_argument("No reference points set. Call setReferencePoints() first.");
        }

        if (left_points.empty()) {
            throw std::invalid_argument("Left point set is empty");
        }
        if (left_points.size() > params_.max_points) {
            throw std::invalid_argument("Point count exceeds max_points");
        }
        validatePoints(left_points, "left");

        stats_ = MarkerMatchStats();
        stats_.used_precomputed = true;

        result.disparities.assign(left_points.size(), std::numeric_limits<float>::quiet_NaN());
        result.valid_flags.assign(left_points.size(), static_cast<uint8_t>(0));
        result.confidence.assign(left_points.size(), 0.0f);
        result.centerMatches.assign(left_points.size(), -1);

        auto sort_start = std::chrono::high_resolution_clock::now();

        std::vector<IndexPoint> left_sorted(left_points.size());
        for (size_t i = 0; i < left_points.size(); ++i) {
            left_sorted[i] = IndexPoint(i, left_points[i]);
        }
        std::sort(left_sorted.begin(), left_sorted.end());

        auto sort_end = std::chrono::high_resolution_clock::now();
        stats_.sort_time_ms = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();

        std::vector<uint8_t> left_visited(left_points.size(), 0);
        std::vector<uint8_t> right_visited(precomputed_right_points_.size(), 0);

        float y_tol = calculateAdaptiveTolerance(
            left_points, params_.y_tolerance,
            params_.density_threshold_high, params_.density_threshold_low,
            params_.dense_tolerance_scale, params_.sparse_tolerance_scale);

        size_t required_buffer_size = std::max(left_points.size(), precomputed_right_points_.size());
        if (left_window_buffer_.size() < required_buffer_size) {
            resizeBuffers(required_buffer_size);
        }

        auto match_start = std::chrono::high_resolution_clock::now();

        size_t matched_count = 0;
        size_t ambiguous_count = 0;
        size_t unvisited_count = 0;

        neighbor_disparities_buffer_.clear();
        neighbor_disparities_buffer_.reserve(left_points.size());

        for (size_t i = 0; i < left_sorted.size(); ++i) {
            const IndexPoint& current_left = left_sorted[i];

            if (left_visited[current_left.original_idx]) {
                continue;
            }

            float center_y = current_left.y;
            float low_y = center_y - y_tol;
            float high_y = center_y + y_tol;

            size_t left_count = 0;
            left_window_buffer_[left_count++] = current_left.original_idx;

            for (size_t k = i + 1; k < left_sorted.size(); ++k) {
                if (left_sorted[k].y > high_y) break;
                uint32_t orig_idx = left_sorted[k].original_idx;
                if (!left_visited[orig_idx]) {
                    if (left_count >= left_window_buffer_.size())
                        throw std::runtime_error("Left window buffer overflow");
                    left_window_buffer_[left_count++] = orig_idx;
                }
            }

            size_t right_count = 0;

            bool use_yzone = right_yzone_index_.isValid();
            int low_bucket = -1, high_bucket = -1;

            if (use_yzone) {
                low_bucket = right_yzone_index_.getBucketIndex(low_y);
                high_bucket = right_yzone_index_.getBucketIndex(high_y);
                if (low_bucket < 0 || high_bucket < 0) {
                    use_yzone = false;
                }
            }

            if (use_yzone) {
                for (int b = low_bucket; b <= high_bucket && b < right_yzone_index_.num_buckets; ++b) {
                    size_t start = right_yzone_index_.bucket_starts[b];
                    size_t end = right_yzone_index_.bucket_ends[b];

                    for (size_t j = start; j < end; ++j) {
                        const auto& pt = precomputed_right_sorted_[j];
                        if (pt.y < low_y) continue;
                        if (pt.y > high_y) break;

                        if (!right_visited[pt.original_idx]) {
                            if (right_count >= right_window_buffer_.size())
                                throw std::runtime_error("Right window buffer overflow");
                            right_window_buffer_[right_count++] = pt.original_idx;
                        }
                    }
                }
            } else {
                auto lower_it = std::lower_bound(
                    precomputed_right_sorted_.begin(),
                    precomputed_right_sorted_.end(), low_y,
                    [](const IndexPoint& a, float val) { return a.y < val; });

                for (auto it = lower_it; it != precomputed_right_sorted_.end(); ++it) {
                    if (it->y > high_y) break;
                    uint32_t orig_idx = it->original_idx;
                    if (!right_visited[orig_idx]) {
                        if (right_count >= right_window_buffer_.size())
                            throw std::runtime_error("Right window buffer overflow");
                        right_window_buffer_[right_count++] = orig_idx;
                    }
                }
            }

            size_t N_L = left_count;
            size_t N_R = right_count;

            for (size_t j = 0; j < left_count; ++j) {
                left_visited[left_window_buffer_[j]] = 1;
            }
            for (size_t j = 0; j < right_count; ++j) {
                right_visited[right_window_buffer_[j]] = 1;
            }

            if (N_L == 1 && N_R == 1) {
                uint32_t right_orig_idx = static_cast<uint32_t>(right_window_buffer_[0]);
                const auto& matched_right_pt = precomputed_right_points_[right_orig_idx];

                float disparity = current_left.x - matched_right_pt.x;

                const float* neighbor_ptr = neighbor_disparities_buffer_.empty() ? nullptr :
                                            neighbor_disparities_buffer_.data();
                size_t neighbor_count = neighbor_disparities_buffer_.size();

                float conf = calculateConfidence(
                    center_y, matched_right_pt.y, y_tol,
                    disparity, neighbor_ptr, neighbor_count, params_.y_tolerance);

                result.disparities[current_left.original_idx] = disparity;
                result.valid_flags[current_left.original_idx] = 1;
                result.confidence[current_left.original_idx] = conf;
                result.centerMatches[current_left.original_idx] = static_cast<int>(right_orig_idx);

                neighbor_disparities_buffer_.push_back(disparity);
                matched_count++;
            } else if (N_L > 1 || N_R > 1) {
                ambiguous_count += N_L;
            }
        }

        for (size_t i = 0; i < left_visited.size(); ++i) {
            if (!left_visited[i]) unvisited_count++;
        }

        auto match_end = std::chrono::high_resolution_clock::now();
        stats_.match_time_ms = std::chrono::duration<double, std::milli>(match_end - match_start).count();

        stats_.num_threads_used = 1;

        collectMatchStatistics(stats_, matched_count, ambiguous_count, unvisited_count,
                               result.disparities, result.valid_flags, result.confidence);

        auto total_end = std::chrono::high_resolution_clock::now();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        result.success = true;
        result.message = "Matching with reference completed";
        result.qualityFlag = calib::QualityFlag::Normal;

        if (stats_.match_rate < 0.3f && stats_.total_points > 5) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Low match rate: " + std::to_string(static_cast<int>(stats_.match_rate * 100)) + "%";
        }
    }

    void setReferencePoints(const std::vector<cv::Point2f>& right_points) {
        if (right_points.empty())
            throw std::invalid_argument("Reference point set is empty");
        if (right_points.size() > params_.max_points)
            throw std::invalid_argument("Point count exceeds max_points");
        validatePoints(right_points, "right");

        precomputed_right_points_ = right_points;

        precomputed_right_sorted_.resize(right_points.size());
        for (size_t i = 0; i < right_points.size(); ++i) {
            precomputed_right_sorted_[i] = IndexPoint(i, right_points[i]);
        }
        std::sort(precomputed_right_sorted_.begin(), precomputed_right_sorted_.end());

        buildYZoneIndex(precomputed_right_sorted_, right_yzone_index_, params_.y_tolerance);

        has_precomputed_ = true;
    }

    void clearReferencePoints() {
        precomputed_right_sorted_.clear();
        precomputed_right_points_.clear();
        right_yzone_index_.bucket_starts.clear();
        right_yzone_index_.bucket_ends.clear();
        right_yzone_index_.num_buckets = 0;
        has_precomputed_ = false;
    }

    void Warmup(int maxPointCount) {
        warmup_maxPoints_ = maxPointCount;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxPointCount={}", maxPointCount);
    }

    void SetParams(const MarkerMatchCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;

        if (left_window_buffer_.size() < params_.prealloc_buffer_size) {
            resizeBuffers(params_.prealloc_buffer_size);
        }
    }
};

// ============================================================
// MarkerMatchCPU public API
// ============================================================
MarkerMatchCPU::MarkerMatchCPU(const MarkerMatchCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("MarkerMatchCPU initialized (yTolerance={}, maxPoints={})",
                   params.y_tolerance, params.max_points);
}

MarkerMatchCPU::~MarkerMatchCPU() = default;

void MarkerMatchCPU::Destroy() { }

MarkerMatchCPUResult MarkerMatchCPU::Execute(const std::vector<cv::Point2f>& left_points,
                           const std::vector<cv::Point2f>& right_points) {
    CALIB_LOG_DEBUG("match() called: left={}, right={}", left_points.size(), right_points.size());
    MarkerMatchCPUResult result = pImpl_->ExecuteImpl(left_points, right_points);
    result.statistics = pImpl_->stats_;
    return result;
}

void MarkerMatchCPU::SetReferencePoints(const std::vector<cv::Point2f>& right_points) {
    CALIB_LOG_INFO("setReferencePoints() called: count={}", right_points.size());
    pImpl_->setReferencePoints(right_points);
}

MarkerMatchCPUResult MarkerMatchCPU::MatchWithReference(const std::vector<cv::Point2f>& left_points) {
    CALIB_LOG_DEBUG("matchWithReference() called: left={}", left_points.size());
    MarkerMatchCPUResult result;
    pImpl_->ExecuteWithReferenceImpl(left_points, result);
    result.statistics = pImpl_->stats_;
    return result;
}

void MarkerMatchCPU::ClearReferencePoints() noexcept {
    pImpl_->clearReferencePoints();
}

bool MarkerMatchCPU::HasReferencePoints() const noexcept {
    return pImpl_->has_precomputed_;
}

void MarkerMatchCPU::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
}

void MarkerMatchCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

void MarkerMatchCPU::SetParams(const MarkerMatchCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const MarkerMatchCPUParams& MarkerMatchCPU::GetParams() const {
    return pImpl_->params_;
}

const MarkerMatchStats& MarkerMatchCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}

void MarkerMatchCPU::ResetStatistics() noexcept {
    pImpl_->stats_ = MarkerMatchStats();
}

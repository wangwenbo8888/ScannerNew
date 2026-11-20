// artifacts.cpp — 中间产物落盘/再载入实现
#include "artifacts.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

namespace scr {

bool savePng(const std::string& path, const cv::Mat& img) {
    try {
        return cv::imwrite(path, img);
    } catch (...) {
        return false;
    }
}

bool saveMaskOverlayPng(const std::string& path, const cv::Mat& gray,
                        const cv::Mat& mask, cv::Scalar color) {
    try {
        cv::Mat color8c;
        if (gray.channels() == 1)
            cv::cvtColor(gray, color8c, cv::COLOR_GRAY2BGR);
        else
            color8c = gray.clone();
        color8c.setTo(color, mask);
        return cv::imwrite(path, color8c);
    } catch (...) {
        return false;
    }
}

namespace {
template <typename P>
bool saveCsvP2Impl(const std::string& path, const std::vector<P>& pts) {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "x,y\n" << std::fixed << std::setprecision(6);
    for (const auto& p : pts) ofs << p.x << ',' << p.y << '\n';
    return true;
}
} // namespace

bool saveCsvP2f(const std::string& path, const std::vector<cv::Point2f>& pts) {
    return saveCsvP2Impl(path, pts);
}
bool saveCsvP2d(const std::string& path, const std::vector<cv::Point2d>& pts) {
    return saveCsvP2Impl(path, pts);
}

bool loadCsvP2f(const std::string& path, std::vector<cv::Point2f>& pts) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string line;
    if (!std::getline(ifs, line)) return false;  // 表头
    pts.clear();
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string xs, ys;
        if (!std::getline(ss, xs, ',') || !std::getline(ss, ys, ',')) return false;
        try {
            pts.emplace_back(std::stof(xs), std::stof(ys));
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool saveCsvEdgePoints(const std::string& path, const std::vector<EdgePointCsv>& pts) {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "x,y,angle,amplitude,pixelX,pixelY\n" << std::fixed << std::setprecision(6);
    for (const auto& p : pts)
        ofs << p.x << ',' << p.y << ',' << p.angle << ',' << p.amplitude
            << ',' << p.pixelX << ',' << p.pixelY << '\n';
    return true;
}

bool saveGpuMatP2(const std::string& path, const cv::cuda::GpuMat& d_pts,
                  cv::cuda::Stream& stream) {
    try {
        // ROI 视图先物化（copyTo → 连续缓冲），规避视图 step/pitch 差异
        cv::cuda::GpuMat contiguous;
        d_pts.copyTo(contiguous, stream);
        cv::Mat host;
        contiguous.download(host, stream);
        stream.waitForCompletion();
        if (host.empty() || host.type() != CV_32FC2) {
            // 非法/空视图：写空表头保产物存在
            ensureCsvHeader(path, "x,y");
            return true;
        }
        std::vector<cv::Point2f> pts(host.begin<cv::Point2f>(),
                                     host.end<cv::Point2f>());
        return saveCsvP2f(path, pts);
    } catch (const std::exception& e) {
        std::cerr << "[artifacts] saveGpuMatP2 异常: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[artifacts] saveGpuMatP2 未知异常" << std::endl;
        return false;
    }
}

bool loadGpuMatP2(const std::string& path, cv::cuda::GpuMat& d_pts,
                  cv::cuda::Stream& stream) {
    std::vector<cv::Point2f> pts;
    if (!loadCsvP2f(path, pts) || pts.empty()) return false;
    try {
        cv::Mat host(pts, true);  // N×1 CV_32FC2
        d_pts.upload(host, stream);
        stream.waitForCompletion();
        return true;
    } catch (...) {
        return false;
    }
}

namespace {
template <typename P>
bool savePlyImpl(const std::string& path, const std::vector<P>& pts) {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
        << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        << std::fixed << std::setprecision(4);
    for (const auto& p : pts) ofs << p.x << ' ' << p.y << ' ' << p.z << '\n';
    return true;
}
} // namespace

bool savePlyXYZf(const std::string& path, const std::vector<cv::Point3f>& pts) {
    return savePlyImpl(path, pts);
}
bool savePlyXYZd(const std::string& path, const std::vector<cv::Point3d>& pts) {
    return savePlyImpl(path, pts);
}

void appendCsvRow(const std::string& path, const std::string& headerOnce,
                  const std::string& row) {
    // 文件不存在或首行为空 → 先写表头；随后追加行（单进程顺序追加语义）
    ensureCsvHeader(path, headerOnce);
    std::ofstream ofs(path, std::ios::app);
    if (ofs) ofs << row << '\n';
}

void ensureCsvHeader(const std::string& path, const std::string& header) {
    std::ifstream chk(path);
    std::string firstLine;
    if (!chk || !std::getline(chk, firstLine) || firstLine.empty()) {
        std::ofstream ofs(path);
        if (ofs) ofs << header << '\n';
    }
}

bool saveJson(const std::string& path, const nlohmann::json& j) {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << j.dump(2) << '\n';
    return true;
}

} // namespace scr

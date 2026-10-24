/**
 * @file mask_sep_cli.cpp
 * @brief 激光线与标记点掩膜分离算子离线调试工具（PNG 输入输出）
 *
 * Debug 自建 OpenCV 无 imgcodecs，故 PNG 解码/编码走 Windows WIC。
 *
 * 用法：mask_sep_cli <in.png> <outDir>
 *          [threshold noiseErodeSize laserKillSize markerKillSize
 *           laserEdgeDilateSize markingEdgeDilateSize]
 * 输出：<outDir>\{laser,marking,combined}_mask.png（8UC1 二值）
 *      stdout：连通域统计（数量/宽度/最大域 bbox/质心）＋分步耗时
 */

#include "laser_markingpoint_mask_separation_cuda.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}

bool loadGrayPng(const std::wstring& path, cv::Mat& out) {
    ComPtr<IWICImagingFactory> factory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(factory.GetAddressOf()));
    if (FAILED(hr)) return false;
    ComPtr<IWICBitmapDecoder> decoder;
    hr = factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, decoder.GetAddressOf());
    if (FAILED(hr)) return false;
    ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, frame.GetAddressOf());
    if (FAILED(hr)) return false;
    ComPtr<IWICBitmapSource> src;
    if (FAILED(frame.As(&src))) return false;
    ComPtr<IWICBitmapSource> gray;
    hr = WICConvertBitmapSource(GUID_WICPixelFormat8bppGray, src.Get(), gray.GetAddressOf());
    if (FAILED(hr)) return false;
    UINT w = 0, h = 0;
    if (FAILED(gray->GetSize(&w, &h)) || w == 0 || h == 0) return false;
    out.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
    hr = gray->CopyPixels(nullptr, static_cast<UINT>(out.step[0]),
                          static_cast<UINT>(out.total()), out.data);
    return SUCCEEDED(hr);
}

bool saveGrayPng(const std::wstring& path, const cv::Mat& m) {
    ComPtr<IWICImagingFactory> factory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(factory.GetAddressOf()));
    if (FAILED(hr)) return false;
    ComPtr<IWICStream> stream;
    if (FAILED(factory->CreateStream(stream.GetAddressOf()))) return false;
    if (FAILED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return false;
    ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.GetAddressOf())))
        return false;
    if (FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;
    ComPtr<IWICBitmapFrameEncode> frameEnc;
    ComPtr<IPropertyBag2> props;
    if (FAILED(encoder->CreateNewFrame(frameEnc.GetAddressOf(), props.GetAddressOf())))
        return false;
    if (FAILED(frameEnc->Initialize(props.Get()))) return false;
    if (FAILED(frameEnc->SetSize(m.cols, m.rows))) return false;
    WICPixelFormatGUID fmt = GUID_WICPixelFormat8bppGray;
    if (FAILED(frameEnc->SetPixelFormat(&fmt))) return false;
    if (FAILED(frameEnc->WritePixels(static_cast<UINT>(m.rows), static_cast<UINT>(m.step[0]),
                                     static_cast<UINT>(m.total()), m.data)))
        return false;
    if (FAILED(frameEnc->Commit())) return false;
    return SUCCEEDED(encoder->Commit());
}

void report(const std::string& name, const cv::Mat& mask) {
    const int n = cv::countNonZero(mask);
    std::cout << name << ": nonzero=" << n << " ("
              << (100.0 * n / static_cast<double>(mask.total())) << "%)\n";
    if (n == 0) return;

    cv::Mat labels, stats, centroids;
    const int comps = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    std::cout << "  components=" << comps - 1 << "\n";

    std::vector<int> order(comps - 1);
    for (int i = 0; i < comps - 1; ++i) order[i] = i + 1;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA);
    });

    const int best = order[0];
    std::cout << "  largest(area=" << stats.at<int>(best, cv::CC_STAT_AREA)
              << " w=" << stats.at<int>(best, cv::CC_STAT_WIDTH)
              << " h=" << stats.at<int>(best, cv::CC_STAT_HEIGHT)
              << " bbox=[" << stats.at<int>(best, cv::CC_STAT_LEFT) << ","
              << stats.at<int>(best, cv::CC_STAT_TOP) << "]\n";

    if (comps - 1 <= 40) {
        std::vector<int> widths;
        widths.reserve(comps - 1);
        for (int i = 0; i < comps - 1; ++i) {
            const int id = order[i];
            const int w = stats.at<int>(id, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(id, cv::CC_STAT_HEIGHT);
            widths.push_back(std::min(w, h));
            std::cout << "  #" << i << " area=" << stats.at<int>(id, cv::CC_STAT_AREA) << " w=" << w
                      << " h=" << h << " at(" << stats.at<int>(id, cv::CC_STAT_LEFT) << ","
                      << stats.at<int>(id, cv::CC_STAT_TOP) << ") cx="
                      << static_cast<int>(centroids.at<double>(id, 0)) << " cy="
                      << static_cast<int>(centroids.at<double>(id, 1)) << "\n";
        }
        std::sort(widths.begin(), widths.end());
        std::cout << "  min-width median=" << widths[widths.size() / 2] << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: mask_sep_cli <in.png> <outDir> [threshold noiseErodeSize "
                     "laserKillSize markerKillSize laserEdgeDilateSize markingEdgeDilateSize]\n";
        return 2;
    }
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        std::cerr << "COM init failed\n";
        return 2;
    }

    cv::Mat gray;
    if (!loadGrayPng(widen(argv[1]), gray) || gray.empty()) {
        std::cerr << "load png failed: " << argv[1] << "\n";
        return 2;
    }
    std::cout << "input: " << argv[1] << " " << gray.cols << "x" << gray.rows << "\n";

    calib::LaserMarkingSeparationParams p;
    if (argc > 3) p.threshold = std::atoi(argv[3]);
    if (argc > 4) p.noiseErodeSize = std::atoi(argv[4]);
    if (argc > 5) p.laserKillSize = std::atoi(argv[5]);
    if (argc > 6) p.markerKillSize = std::atoi(argv[6]);
    if (argc > 7) p.laserEdgeDilateSize = std::atoi(argv[7]);
    if (argc > 8) p.markingEdgeDilateSize = std::atoi(argv[8]);
    try {
        p.validate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    std::cout << "params: thr=" << p.threshold << " noise=" << p.noiseErodeSize
              << " laserKill=" << p.laserKillSize << " markerKill=" << p.markerKillSize
              << " laserDilate=" << p.laserEdgeDilateSize
              << " markingDilate=" << p.markingEdgeDilateSize << "\n";

    calib::LaserMarkingSeparationCUDA separator(p);
    separator.Warmup(gray.rows, gray.cols);

    calib::LaserMarkingSeparationResult r = separator.Execute(gray);
    if (!r.success) {
        std::cerr << "separation failed: " << r.message << "\n";
        return 1;
    }

    cv::Mat laser, marking, combined;
    r.d_laserMask->download(laser);
    r.d_markingPointMask->download(marking);
    r.d_combinedMask->download(combined);

    const std::string outDir = argv[2];
    saveGrayPng(widen(outDir + "\\laser_mask.png"), laser);
    saveGrayPng(widen(outDir + "\\marking_mask.png"), marking);
    saveGrayPng(widen(outDir + "\\combined_mask.png"), combined);

    report("laser  ", laser);
    report("marking", marking);
    report("combined", combined);

    const calib::MaskSeparationTimings& t = r.timings;
    std::cout << "timings(ms): gauss+thr=" << t.step1_gaussian_threshold_ms
              << " noiseErode=" << t.step2_noise_erode_ms << " laser=" << t.step3_laser_extract_ms
              << " marking=" << t.step4_marking_extract_ms << " dilate=" << t.step5_edge_dilate_ms
              << " upload=" << t.upload_ms << " total=" << t.total_pipeline_ms << "\n";
    std::cout << "saved: " << outDir << "\\{laser,marking,combined}_mask.png\n";
    return 0;
}

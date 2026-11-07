/**
 * @file cmtt_table_loader.cpp
 * @brief CMTT sidecar → 单档 LaserPlaneMapTempTable 装载器实现
 *
 * 实现链: 整文件定长单读（seekg(end)/tellg → 单次 read）→ CmttReader::load
 * （magic/版本/索引边界校验）→ nearestTier（等距偏低温）→ tierBytes（CRC 懒校验后
 * 拷出）→ 档温走 CmttReader::tierTemperature（索引项直存值; 生产写方的
 * tempBaseX10 + (idx−nHalf)·tempStepX10 对称展开只是等价推论）→
 * CurveMapTable::Load → 逐行 EnumerateRow 两段式容量展开 →
 * Vec4f(uL·rowStep, row·rowStep, uR·rowStep, lid) → 单档 LaserPlaneMapTempTable。
 * 全函数体 try/catch 兜底（bad_alloc/cv::Exception → err 文案, 不外抛）。
 */

#include "cmtt_table_loader.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <vector>

#include "calibration/laser_calib/curve_map/curve_map.h"
#include "calibration/laser_calib/curve_map_temp_table/cmtt_container.h"

namespace calib {

std::shared_ptr<const LaserPlaneMapTempTable>
LoadTempTableFromCmtt(const std::string& cmttPath, double tempC, std::string& err,
                      int* tierIdx, double* tierTemp, float* rowStep) {
    err.clear();
    try {
        std::ifstream ifs(cmttPath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            err = "CmttTableLoader: cannot open file: " + cmttPath;
            return nullptr;
        }
        const std::streamoff size = ifs.tellg();
        if (size < 0) {
            err = "CmttTableLoader: cannot determine file size: " + cmttPath;
            return nullptr;
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        if (size > 0) {
            ifs.seekg(0, std::ios::beg);
            ifs.read(reinterpret_cast<char*>(bytes.data()),
                     static_cast<std::streamsize>(size));
            if (ifs.gcount() != static_cast<std::streamsize>(size)) {
                err = "CmttTableLoader: short read: " + cmttPath;
                return nullptr;
            }
        }
        if (bytes.empty()) {
            err = "CmttTableLoader: file is empty: " + cmttPath;
            return nullptr;
        }

        CmttReader reader;
        if (!reader.load(bytes.data(), bytes.size())) {
            err = "CmttTableLoader: container load rejected (bad magic/version/index): "
                  + cmttPath;
            return nullptr;
        }

        const auto idxOpt = reader.nearestTier(tempC);
        if (!idxOpt) {
            err = "CmttTableLoader: no tiers in container: " + cmttPath;
            return nullptr;
        }
        const size_t idx = *idxOpt;

        const auto blobOpt = reader.tierBytes(idx);
        if (!blobOpt) {
            err = "CmttTableLoader: tier " + std::to_string(idx)
                  + " payload unavailable (missing or crc mismatch): " + cmttPath;
            return nullptr;
        }

        const double tierTempC = reader.tierTemperature(idx);

        CurveMapTable cmt;
        std::string loadErr;
        if (!cmt.Load(blobOpt->data(), blobOpt->size(), loadErr)) {
            err = "CmttTableLoader: tier " + std::to_string(idx)
                  + " CMPU load failed: " + loadErr;
            return nullptr;
        }

        const float step = cmt.rowStep();
        std::vector<cv::Vec4f> rows;
        std::vector<CurveMapEntry> buf(64);
        for (uint32_t row = 0; row < cmt.rowCount(); ++row) {
            int n = 0;
            while ((n = cmt.EnumerateRow(static_cast<int>(row), buf.data(),
                                         static_cast<int>(buf.size()))) < 0) {
                buf.resize(buf.size() * 2);
            }
            for (int i = 0; i < n; ++i) {
                const CurveMapEntry& e = buf[static_cast<size_t>(i)];
                rows.emplace_back(e.uL * step, e.row * step, e.uR * step,
                                  static_cast<float>(e.lid));
            }
        }

        if (rows.empty() || rows.size() != cmt.entryCount()) {
            err = "CmttTableLoader: tier " + std::to_string(idx)
                  + " entry decode mismatch (enumerated " + std::to_string(rows.size())
                  + ", header " + std::to_string(cmt.entryCount()) + ")";
            return nullptr;
        }

        auto table = std::make_shared<LaserPlaneMapTempTable>();
        LaserPlaneMap planeMap;
        planeMap.temperature = tierTempC;
        planeMap.leftToRightMap =
            cv::Mat(static_cast<int>(rows.size()), 1, CV_32FC4, rows.data()).clone();
        planeMap.totalPairs = static_cast<int>(rows.size());
        table->referenceTemperature = tierTempC;
        table->tempStep = reader.header().tempStepX10 / 10.0;
        table->table[tierTempC] = std::move(planeMap);

        if (tierIdx) *tierIdx = static_cast<int>(idx);
        if (tierTemp) *tierTemp = tierTempC;
        if (rowStep) *rowStep = cmt.rowStep();
        return table;
    } catch (const std::exception& e) {
        err = "CmttTableLoader: unexpected exception: " + std::string(e.what());
        return nullptr;
    } catch (...) {
        err = "CmttTableLoader: unknown exception";
        return nullptr;
    }
}

} // namespace calib

#include "hnswlib.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

const size_t kDim = 16;
const size_t kCount = 128;
const hnswlib::labeltype kProbeLabel = 37;

std::vector<float> makeData(size_t count) {
    std::vector<float> data(count * kDim);
    for (size_t row = 0; row < count; ++row) {
        for (size_t col = 0; col < kDim; ++col) {
            const int value =
                static_cast<int>((row * 37 + col * 19) % 1000);
            data[row * kDim + col] =
                static_cast<float>(value) / 1000.0f;
        }
    }
    return data;
}

std::vector<float> probeData() {
    std::vector<float> data(kDim);
    for (size_t col = 0; col < kDim; ++col) {
        const int value =
            static_cast<int>((kProbeLabel * 37 + col * 19) % 1000);
        data[col] = static_cast<float>(value) / 1000.0f;
    }
    return data;
}

int buildIndex(const std::string &path) {
    hnswlib::L2Space space(static_cast<int>(kDim));
    hnswlib::HierarchicalNSW<float> index(&space, kCount, 8, 80, 100);
    index.setCompatibilitySpaceId(hnswlib::INDEX_SPACE_L2);
    const std::vector<float> data = makeData(kCount);
    for (size_t row = 0; row < kCount; ++row)
        index.addPoint(data.data() + row * kDim, row);
    index.saveIndex(path);
    std::cout << "BUILD_OK format=" << hnswlib::INDEX_FORMAT_VERSION
              << " reader=" << hnswlib::INDEX_READER_VERSION
              << " path=" << path << " count=" << kCount << std::endl;
    return 0;
}

int loadIndex(const std::string &path) {
    try {
        hnswlib::L2Space space(static_cast<int>(kDim));
        hnswlib::HierarchicalNSW<float> index(&space, path);
        const std::vector<float> query = probeData();
        const auto result = index.searchKnn(query.data(), 1);
        const hnswlib::labeltype label =
            result.empty() ? static_cast<hnswlib::labeltype>(-1)
                           : result.top().second;
        if (label != kProbeLabel) {
            std::cerr << "LOAD_QUERY_MISMATCH expected=" << kProbeLabel
                      << " actual=" << label << " path=" << path << std::endl;
            return 3;
        }
        std::cout << "LOAD_OK format=" << hnswlib::INDEX_FORMAT_VERSION
                  << " reader=" << hnswlib::INDEX_READER_VERSION
                  << " path=" << path << " count="
                  << index.cur_element_count.load() << " query_label="
                  << label << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cout << "LOAD_REJECTED reader=" << hnswlib::INDEX_READER_VERSION
                  << " path=" << path << " reason=" << error.what()
                  << std::endl;
        return 2;
    }
}

int inspectIndex(const std::string &path, uint32_t reader_version) {
    const hnswlib::IndexCompatibilityReport report =
        hnswlib::inspectIndexFile(path, hnswlib::INDEX_SPACE_UNKNOWN, 0, 0, 0,
                                  reader_version);
    std::cout << "INSPECT path=" << path
              << " status=" << report.status
              << " format=" << report.metadata.format_version
              << " min_reader=" << report.metadata.min_reader_version
              << " loadable=" << (report.loadable ? "yes" : "no")
              << " reason=\"" << report.reason << "\"";
    for (size_t i = 0; i < report.warnings.size(); ++i)
        std::cout << " warning=\"" << report.warnings[i] << "\"";
    std::cout << std::endl;
    return 0;
}

int scanIndexes(int argc, char **argv) {
    size_t upgradeable = 0;
    size_t rebuild = 0;
    for (int i = 2; i < argc; ++i) {
        const hnswlib::IndexCompatibilityReport report =
            hnswlib::inspectIndexFile(argv[i]);
        std::cout << "SCAN path=" << argv[i]
                  << " category="
                  << (report.loadable ? "upgradeable" : "rebuild_required")
                  << " status=" << report.status
                  << " reason=\"" << report.reason << "\"" << std::endl;
        if (report.loadable)
            ++upgradeable;
        else
            ++rebuild;
    }
    std::cout << "SCAN_SUMMARY upgradeable=" << upgradeable
              << " rebuild_required=" << rebuild << std::endl;
    return 0;
}

void usage(const char *program) {
    std::cerr << "usage: " << program
              << " build <path> | load <path> | inspect <path> "
                 "[reader-version] | scan <path>..." << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "build")
        return buildIndex(argv[2]);
    if (mode == "load")
        return loadIndex(argv[2]);
    if (mode == "inspect") {
        const uint32_t reader_version =
            argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3]))
                     : hnswlib::INDEX_READER_VERSION;
        return inspectIndex(argv[2], reader_version);
    }
    if (mode == "scan")
        return scanIndexes(argc, argv);
    usage(argv[0]);
    return 1;
}

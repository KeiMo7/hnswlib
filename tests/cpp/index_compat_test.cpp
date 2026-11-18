#include "hnswlib.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

const size_t kDim = 12;
const size_t kCount = 64;

std::vector<char> readBytes(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!bytes.empty())
        input.read(bytes.data(), size);
    return bytes;
}

void writeBytes(const std::string &path, const std::vector<char> &bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!bytes.empty())
        output.write(bytes.data(), bytes.size());
}

void putU32(std::vector<char> *bytes, size_t offset, uint32_t value) {
    (*bytes)[offset + 0] = static_cast<char>(value & 0xffu);
    (*bytes)[offset + 1] = static_cast<char>((value >> 8) & 0xffu);
    (*bytes)[offset + 2] = static_cast<char>((value >> 16) & 0xffu);
    (*bytes)[offset + 3] = static_cast<char>((value >> 24) & 0xffu);
}

std::vector<float> makeData() {
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>((i * 17) % 251) / 251.0f;
    return data;
}

bool expectQuery(const std::string &path, const std::vector<float> &data) {
    hnswlib::L2Space space(kDim);
    hnswlib::HierarchicalNSW<float> index(&space, path);
    for (size_t row = 0; row < kCount; ++row) {
        const auto result =
            index.searchKnn(data.data() + row * kDim, 1);
        if (result.empty() || result.top().second != row)
            return false;
    }
    return true;
}

int fail(const std::string &message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main() {
    const std::string base =
        "hnsw_index_compat_test_" +
        std::to_string(static_cast<unsigned long long>(
            reinterpret_cast<uintptr_t>(&makeData)));
    const std::string current_path = base + "_current.bin";
    const std::string legacy_path = base + "_legacy.bin";
    const std::string truncated_path = base + "_truncated.bin";
    const std::string swapped_path = base + "_swapped.bin";
    const std::vector<float> data = makeData();

    {
        hnswlib::L2Space space(kDim);
        hnswlib::HierarchicalNSW<float> index(&space, kCount, 8, 40, 100);
        index.setCompatibilitySpaceId(hnswlib::INDEX_SPACE_L2);
        for (size_t row = 0; row < kCount; ++row)
            index.addPoint(data.data() + row * kDim, row);
        index.saveIndex(current_path);
    }

    const hnswlib::IndexCompatibilityReport current_report =
        hnswlib::inspectIndexFile(current_path, "l2", kDim);
    if (!current_report.loadable || current_report.legacy)
        return fail("same-version index should be loadable");
    if (!expectQuery(current_path, data))
        return fail("same-version query results changed");

    const std::vector<char> current_bytes = readBytes(current_path);
    if (hnswlib::INDEX_FORMAT_VERSION >= 2) {
        const size_t header_end =
            hnswlib::legacyHeaderSize() +
            hnswlib::INDEX_COMPAT_HEADER_SIZE;
        const size_t footer_start =
            current_bytes.size() - hnswlib::INDEX_COMPAT_FOOTER_SIZE;
        std::vector<char> legacy_bytes(current_bytes.begin(),
                                       current_bytes.begin() +
                                           hnswlib::legacyHeaderSize());
        legacy_bytes.insert(legacy_bytes.end(),
                            current_bytes.begin() + header_end,
                            current_bytes.begin() + footer_start);
        writeBytes(legacy_path, legacy_bytes);
    } else {
        writeBytes(legacy_path, current_bytes);
    }
    const hnswlib::IndexCompatibilityReport legacy_report =
        hnswlib::inspectIndexFile(legacy_path, "l2", kDim);
    if (!legacy_report.loadable ||
        (!legacy_report.legacy &&
         legacy_report.metadata.format_version !=
             hnswlib::INDEX_FORMAT_LEGACY))
        return fail("legacy index should remain loadable");
    if (!expectQuery(legacy_path, data))
        return fail("legacy index query results changed");

    {
        hnswlib::L2Space wrong_space(kDim + 1);
        try {
            hnswlib::HierarchicalNSW<float> wrong(&wrong_space, current_path);
            return fail("dimension conflict was not rejected");
        } catch (const std::exception &error) {
            if (std::string(error.what()).find("parameter conflict") ==
                std::string::npos)
                return fail("dimension conflict lacked explicit reason");
        }
    }

    std::vector<char> truncated_bytes = current_bytes;
    truncated_bytes.resize(truncated_bytes.size() - 16);
    writeBytes(truncated_path, truncated_bytes);
    const hnswlib::IndexCompatibilityReport truncated_report =
        hnswlib::inspectIndexFile(truncated_path);
    if (truncated_report.loadable || !truncated_report.truncated)
        return fail("truncated index was not classified for rebuild");
    {
        hnswlib::L2Space space(kDim);
        try {
            hnswlib::HierarchicalNSW<float> truncated(&space, truncated_path);
            return fail("truncated index was not rejected");
        } catch (const std::exception &error) {
            if (std::string(error.what()).find("truncated") ==
                std::string::npos)
                return fail("truncated index error was not explicit");
        }
    }

    if (hnswlib::INDEX_FORMAT_VERSION >= 2) {
        std::vector<char> swapped_bytes = current_bytes;
        const uint32_t opposite_endian =
            hnswlib::hostByteOrder() == hnswlib::INDEX_BYTE_ORDER_LITTLE
                ? hnswlib::INDEX_BYTE_ORDER_BIG
                : hnswlib::INDEX_BYTE_ORDER_LITTLE;
        const size_t header_payload =
            hnswlib::legacyHeaderSize() +
            sizeof(hnswlib::INDEX_COMPAT_HEADER_MAGIC);
        const size_t footer_payload =
            swapped_bytes.size() - hnswlib::INDEX_COMPAT_FOOTER_SIZE +
            sizeof(hnswlib::INDEX_COMPAT_FOOTER_MAGIC);
        putU32(&swapped_bytes, header_payload + 16, opposite_endian);
        putU32(&swapped_bytes, footer_payload + 16, opposite_endian);
        writeBytes(swapped_path, swapped_bytes);
        const hnswlib::IndexCompatibilityReport swapped_report =
            hnswlib::inspectIndexFile(swapped_path);
        if (!swapped_report.endian_mismatch ||
            swapped_report.loadable)
            return fail("byte-order mismatch was not rejected");
    }

    std::remove(current_path.c_str());
    std::remove(legacy_path.c_str());
    std::remove(truncated_path.c_str());
    std::remove(swapped_path.c_str());
    std::cout << "index compatibility tests passed" << std::endl;
    return 0;
}

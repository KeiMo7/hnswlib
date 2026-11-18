#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// The serialized body keeps hnswlib's historical native-POD layout. These
// versions describe the surrounding compatibility envelope, not the graph
// algorithm. A writer that sets HNSWLIB_INDEX_FORMAT_VERSION=1 emits the
// original file byte-for-byte.
#ifndef HNSWLIB_INDEX_FORMAT_VERSION
#define HNSWLIB_INDEX_FORMAT_VERSION 2
#endif

#ifndef HNSWLIB_INDEX_READER_VERSION
#define HNSWLIB_INDEX_READER_VERSION HNSWLIB_INDEX_FORMAT_VERSION
#endif

namespace hnswlib {

static const uint32_t INDEX_FORMAT_LEGACY = 1;
static const uint32_t INDEX_FORMAT_VERSION =
    static_cast<uint32_t>(HNSWLIB_INDEX_FORMAT_VERSION);
static const uint32_t INDEX_READER_VERSION =
    static_cast<uint32_t>(HNSWLIB_INDEX_READER_VERSION);

static const uint32_t INDEX_COMPAT_METADATA_VERSION = 1;
static const uint32_t INDEX_SPACE_UNKNOWN = 0;
static const uint32_t INDEX_SPACE_L2 = 1;
static const uint32_t INDEX_SPACE_IP = 2;
static const uint32_t INDEX_SPACE_COSINE = 3;

static const uint32_t INDEX_BYTE_ORDER_LITTLE = 0x01020304u;
static const uint32_t INDEX_BYTE_ORDER_BIG = 0x04030201u;

static const char INDEX_COMPAT_HEADER_MAGIC[16] = {
    'H', 'N', 'S', 'W', '-', 'I', 'D', 'X',
    '-', 'C', 'O', 'M', 'P', 'A', 'T', '\0'
};
static const char INDEX_COMPAT_FOOTER_MAGIC[16] = {
    'H', 'N', 'S', 'W', '-', 'I', 'D', 'X',
    '-', 'F', 'O', 'O', 'T', 'E', 'R', '\0'
};

// Header/footer payloads use fixed-width little-endian fields. The historical
// 13-field index header and the graph body remain native-POD.
static const size_t INDEX_COMPAT_METADATA_SIZE = 160;
static const size_t INDEX_COMPAT_HEADER_SIZE =
    sizeof(INDEX_COMPAT_HEADER_MAGIC) + INDEX_COMPAT_METADATA_SIZE;
static const size_t INDEX_COMPAT_FOOTER_SIZE =
    sizeof(INDEX_COMPAT_FOOTER_MAGIC) + INDEX_COMPAT_METADATA_SIZE;

inline uint32_t hostByteOrder() {
#if defined(_WIN32)
    return INDEX_BYTE_ORDER_LITTLE;
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return INDEX_BYTE_ORDER_BIG;
#elif defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__)
    return INDEX_BYTE_ORDER_BIG;
#else
    return INDEX_BYTE_ORDER_LITTLE;
#endif
}

inline const char *byteOrderName(uint32_t byte_order) {
    if (byte_order == INDEX_BYTE_ORDER_LITTLE)
        return "little-endian";
    if (byte_order == INDEX_BYTE_ORDER_BIG)
        return "big-endian";
    return "unknown";
}

inline const char *spaceName(uint32_t space_id) {
    switch (space_id) {
        case INDEX_SPACE_L2:
            return "l2";
        case INDEX_SPACE_IP:
            return "ip";
        case INDEX_SPACE_COSINE:
            return "cosine";
        default:
            return "unknown";
    }
}

inline uint32_t spaceIdFromName(const std::string &space_name) {
    if (space_name == "l2")
        return INDEX_SPACE_L2;
    if (space_name == "ip")
        return INDEX_SPACE_IP;
    if (space_name == "cosine")
        return INDEX_SPACE_COSINE;
    return INDEX_SPACE_UNKNOWN;
}

struct IndexFormatMetadata {
    bool present{false};
    uint32_t metadata_version{0};
    uint32_t format_version{INDEX_FORMAT_LEGACY};
    uint32_t min_reader_version{INDEX_FORMAT_LEGACY};
    uint32_t flags{0};
    uint32_t body_byte_order{INDEX_BYTE_ORDER_LITTLE};
    uint32_t size_t_size{0};
    uint32_t pointer_size{0};
    uint32_t max_align{0};
    uint32_t space_id{INDEX_SPACE_UNKNOWN};
    uint32_t dim{0};
    uint32_t distance_type_size{0};
    uint64_t max_elements{0};
    uint64_t element_count{0};
    uint64_t graph_m{0};
    uint64_t ef_construction{0};
    uint64_t header_offset{0};
    uint64_t header_size{0};
    uint64_t body_size{0};
    uint64_t file_size{0};
    uint64_t size_data_per_element{0};
    uint64_t offset_data{0};
    uint64_t label_offset{0};
    uint64_t data_alignment{0};
    uint64_t data_size{0};
};

struct LegacyIndexHeader {
    size_t offset_level0{0};
    size_t max_elements{0};
    size_t element_count{0};
    size_t size_data_per_element{0};
    size_t label_offset{0};
    size_t offset_data{0};
    int max_level{0};
    uint32_t enterpoint_node{0};
    size_t max_m{0};
    size_t max_m0{0};
    size_t graph_m{0};
    double mult{0.0};
    size_t ef_construction{0};
};

struct IndexCompatibilityReport {
    std::string path;
    std::string status{"requires_rebuild"};
    std::string reason;
    bool readable{false};
    bool loadable{false};
    bool upgradeable{false};
    bool requires_rebuild{true};
    bool legacy{false};
    bool truncated{false};
    bool endian_mismatch{false};
    bool layout_mismatch{false};
    bool parameter_conflict{false};
    std::vector<std::string> warnings;
    IndexFormatMetadata metadata;
};

inline size_t legacyHeaderSize() {
    return sizeof(size_t) * 6 + sizeof(int) + sizeof(uint32_t) +
           sizeof(size_t) * 3 + sizeof(double) + sizeof(size_t);
}

inline void appendU32(std::vector<unsigned char> &out, uint32_t value) {
    out.push_back(static_cast<unsigned char>(value & 0xffu));
    out.push_back(static_cast<unsigned char>((value >> 8) & 0xffu));
    out.push_back(static_cast<unsigned char>((value >> 16) & 0xffu));
    out.push_back(static_cast<unsigned char>((value >> 24) & 0xffu));
}

inline void appendU64(std::vector<unsigned char> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xffu));
}

inline uint32_t readU32(const unsigned char *data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

inline uint64_t readU64(const unsigned char *data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    return value;
}

inline std::vector<unsigned char> encodeMetadata(
    const IndexFormatMetadata &metadata) {
    std::vector<unsigned char> out;
    out.reserve(INDEX_COMPAT_METADATA_SIZE);
    appendU32(out, metadata.metadata_version);
    appendU32(out, metadata.format_version);
    appendU32(out, metadata.min_reader_version);
    appendU32(out, metadata.flags);
    appendU32(out, metadata.body_byte_order);
    appendU32(out, metadata.size_t_size);
    appendU32(out, metadata.pointer_size);
    appendU32(out, metadata.max_align);
    appendU32(out, metadata.space_id);
    appendU32(out, metadata.dim);
    appendU32(out, metadata.distance_type_size);
    appendU32(out, 0);
    appendU64(out, metadata.max_elements);
    appendU64(out, metadata.element_count);
    appendU64(out, metadata.graph_m);
    appendU64(out, metadata.ef_construction);
    appendU64(out, metadata.header_offset);
    appendU64(out, metadata.header_size);
    appendU64(out, metadata.body_size);
    appendU64(out, metadata.file_size);
    appendU64(out, metadata.size_data_per_element);
    appendU64(out, metadata.offset_data);
    appendU64(out, metadata.label_offset);
    appendU64(out, metadata.data_alignment);
    appendU64(out, metadata.data_size);
    out.resize(INDEX_COMPAT_METADATA_SIZE, 0);
    return out;
}

inline bool decodeMetadata(const unsigned char *data, size_t size,
                           IndexFormatMetadata *metadata,
                           std::string *error) {
    if (size < INDEX_COMPAT_METADATA_SIZE) {
        if (error)
            *error = "compatibility metadata is truncated";
        return false;
    }
    metadata->present = true;
    metadata->metadata_version = readU32(data + 0);
    metadata->format_version = readU32(data + 4);
    metadata->min_reader_version = readU32(data + 8);
    metadata->flags = readU32(data + 12);
    metadata->body_byte_order = readU32(data + 16);
    metadata->size_t_size = readU32(data + 20);
    metadata->pointer_size = readU32(data + 24);
    metadata->max_align = readU32(data + 28);
    metadata->space_id = readU32(data + 32);
    metadata->dim = readU32(data + 36);
    metadata->distance_type_size = readU32(data + 40);
    metadata->max_elements = readU64(data + 48);
    metadata->element_count = readU64(data + 56);
    metadata->graph_m = readU64(data + 64);
    metadata->ef_construction = readU64(data + 72);
    metadata->header_offset = readU64(data + 80);
    metadata->header_size = readU64(data + 88);
    metadata->body_size = readU64(data + 96);
    metadata->file_size = readU64(data + 104);
    metadata->size_data_per_element = readU64(data + 112);
    metadata->offset_data = readU64(data + 120);
    metadata->label_offset = readU64(data + 128);
    metadata->data_alignment = readU64(data + 136);
    metadata->data_size = readU64(data + 144);
    if (metadata->metadata_version != INDEX_COMPAT_METADATA_VERSION) {
        if (error) {
            std::ostringstream message;
            message << "unsupported compatibility metadata version "
                    << metadata->metadata_version;
            *error = message.str();
        }
        return false;
    }
    return true;
}

inline bool readExact(std::istream &input, char *data, size_t size) {
    input.read(data, static_cast<std::streamsize>(size));
    return input.good() &&
           static_cast<size_t>(input.gcount()) == size;
}

inline bool readCompatHeaderAt(std::istream &input, uint64_t offset,
                               IndexFormatMetadata *metadata,
                               std::string *error) {
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    char magic[sizeof(INDEX_COMPAT_HEADER_MAGIC)];
    if (!readExact(input, magic, sizeof(magic))) {
        if (error)
            *error = "compatibility header is truncated";
        return false;
    }
    if (std::memcmp(magic, INDEX_COMPAT_HEADER_MAGIC, sizeof(magic)) != 0)
        return false;
    unsigned char payload[INDEX_COMPAT_METADATA_SIZE];
    if (!readExact(input, reinterpret_cast<char *>(payload), sizeof(payload))) {
        if (error)
            *error = "compatibility header is truncated";
        return false;
    }
    return decodeMetadata(payload, sizeof(payload), metadata, error);
}

inline bool readCompatFooter(std::istream &input, uint64_t file_size,
                             IndexFormatMetadata *metadata,
                             std::string *error) {
    if (file_size < INDEX_COMPAT_FOOTER_SIZE)
        return false;
    input.clear();
    input.seekg(static_cast<std::streamoff>(
                    file_size - INDEX_COMPAT_FOOTER_SIZE),
                std::ios::beg);
    char magic[sizeof(INDEX_COMPAT_FOOTER_MAGIC)];
    if (!readExact(input, magic, sizeof(magic)))
        return false;
    if (std::memcmp(magic, INDEX_COMPAT_FOOTER_MAGIC, sizeof(magic)) != 0)
        return false;
    unsigned char payload[INDEX_COMPAT_METADATA_SIZE];
    if (!readExact(input, reinterpret_cast<char *>(payload), sizeof(payload))) {
        if (error)
            *error = "compatibility footer is truncated";
        return false;
    }
    return decodeMetadata(payload, sizeof(payload), metadata, error);
}

inline bool readLegacyHeader(std::istream &input, LegacyIndexHeader *header,
                             std::string *error) {
    input.clear();
    input.seekg(0, std::ios::beg);
    if (!readExact(input, reinterpret_cast<char *>(&header->offset_level0),
                   legacyHeaderSize())) {
        if (error)
            *error = "index header is truncated";
        return false;
    }
    return true;
}

inline bool metadataEquals(const IndexFormatMetadata &a,
                           const IndexFormatMetadata &b) {
    return a.format_version == b.format_version &&
           a.min_reader_version == b.min_reader_version &&
           a.body_byte_order == b.body_byte_order &&
           a.size_t_size == b.size_t_size &&
           a.pointer_size == b.pointer_size &&
           a.space_id == b.space_id && a.dim == b.dim &&
           a.distance_type_size == b.distance_type_size &&
           a.max_align == b.max_align &&
           a.max_elements == b.max_elements &&
           a.element_count == b.element_count && a.graph_m == b.graph_m &&
           a.ef_construction == b.ef_construction &&
           a.header_offset == b.header_offset &&
           a.body_size == b.body_size && a.file_size == b.file_size &&
           a.size_data_per_element == b.size_data_per_element &&
           a.offset_data == b.offset_data &&
           a.label_offset == b.label_offset &&
           a.data_alignment == b.data_alignment &&
           a.data_size == b.data_size;
}

struct IndexEnvelope {
    bool has_header{false};
    bool has_footer{false};
    IndexFormatMetadata metadata;
    std::string error;
};

inline IndexEnvelope readIndexEnvelope(std::istream &input,
                                       uint64_t file_size) {
    IndexEnvelope envelope;
    std::string footer_error;
    envelope.has_footer =
        readCompatFooter(input, file_size, &envelope.metadata, &footer_error);
    IndexFormatMetadata header_metadata;
    std::string header_error;
    const uint64_t header_offset =
        envelope.has_footer ? envelope.metadata.header_offset
                            : legacyHeaderSize();
    envelope.has_header = readCompatHeaderAt(input, header_offset,
                                             &header_metadata, &header_error);
    if (!envelope.has_footer && envelope.has_header)
        envelope.metadata = header_metadata;
    if (envelope.has_footer && envelope.has_header) {
        if (!metadataEquals(envelope.metadata, header_metadata))
            envelope.error =
                "compatibility header and footer metadata disagree";
    } else if (envelope.has_footer && !envelope.has_header) {
        envelope.error = header_error.empty()
                             ? "compatibility header is missing"
                             : header_error;
    } else if (!envelope.has_footer && envelope.has_header) {
        envelope.error =
            "compatibility footer is missing; file is truncated";
    }
    return envelope;
}

inline bool checkedAdd(uint64_t a, uint64_t b, uint64_t *result) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        return false;
    *result = a + b;
    return true;
}

inline bool legacyMinimumSize(const LegacyIndexHeader &header,
                              uint64_t *minimum_size) {
    uint64_t level0_size = 0;
    uint64_t link_headers = 0;
    if (header.element_count != 0 &&
        header.size_data_per_element >
            std::numeric_limits<uint64_t>::max() / header.element_count)
        return false;
    level0_size =
        static_cast<uint64_t>(header.element_count) *
        static_cast<uint64_t>(header.size_data_per_element);
    if (header.element_count != 0 &&
        sizeof(uint32_t) >
            std::numeric_limits<uint64_t>::max() / header.element_count)
        return false;
    link_headers =
        static_cast<uint64_t>(header.element_count) * sizeof(uint32_t);
    uint64_t total = 0;
    if (!checkedAdd(legacyHeaderSize(), level0_size, &total) ||
        !checkedAdd(total, link_headers, &total))
        return false;
    *minimum_size = total;
    return true;
}

inline void addAlignmentWarning(const IndexFormatMetadata &metadata,
                                std::vector<std::string> *warnings) {
    const uint32_t reader_max_align =
        static_cast<uint32_t>(alignof(std::max_align_t));
    if (metadata.max_align != 0 && metadata.max_align != reader_max_align) {
        std::ostringstream message;
        message << "layout alignment profile differs: writer max alignment "
                << metadata.max_align << ", reader max alignment "
                << reader_max_align
                << "; body access is byte-compatible but may be unaligned "
                   "and slower on this platform";
        warnings->push_back(message.str());
    }
    const uint32_t data_unit =
        metadata.distance_type_size ? metadata.distance_type_size : 1;
    if ((metadata.offset_data % data_unit) != 0 ||
        (metadata.size_data_per_element % data_unit) != 0) {
        std::ostringstream message;
        message << "vector payload is not naturally aligned for "
                << data_unit << "-byte distance values (offset "
                << metadata.offset_data << ", stride "
                << metadata.size_data_per_element
                << "); loads may be slower on this platform";
        warnings->push_back(message.str());
    }
}

inline void applyParameterChecks(IndexCompatibilityReport *report,
                                 uint32_t expected_space_id,
                                 uint32_t expected_dim,
                                 uint64_t expected_m,
                                 uint64_t expected_ef_construction) {
    const IndexFormatMetadata &metadata = report->metadata;
    if (expected_space_id != INDEX_SPACE_UNKNOWN) {
        if (metadata.present &&
            metadata.space_id != INDEX_SPACE_UNKNOWN &&
            metadata.space_id != expected_space_id) {
            std::ostringstream message;
            message << "index parameter conflict: file space '"
                    << spaceName(metadata.space_id)
                    << "' does not match loader space '"
                    << spaceName(expected_space_id) << "'";
            report->parameter_conflict = true;
            report->requires_rebuild = true;
            report->loadable = false;
            report->status = "requires_rebuild";
            if (!report->reason.empty())
                report->reason += "; ";
            report->reason += message.str();
        } else if (!metadata.present) {
            report->warnings.push_back(
                "legacy index has no space identifier; space compatibility "
                "is inferred from vector size and cannot be verified");
        }
    }
    if (expected_dim != 0 && metadata.dim != 0 &&
        metadata.dim != expected_dim) {
        std::ostringstream message;
        message << "index parameter conflict: file dim " << metadata.dim
                << " does not match loader dim " << expected_dim;
        report->parameter_conflict = true;
        report->requires_rebuild = true;
        report->loadable = false;
        report->status = "requires_rebuild";
        if (!report->reason.empty())
            report->reason += "; ";
        report->reason += message.str();
    }
    if (expected_m != 0 && metadata.graph_m != 0 &&
        metadata.graph_m != expected_m) {
        std::ostringstream message;
        message << "index parameter conflict: file M " << metadata.graph_m
                << " does not match requested M " << expected_m;
        report->parameter_conflict = true;
        report->requires_rebuild = true;
        report->loadable = false;
        report->status = "requires_rebuild";
        if (!report->reason.empty())
            report->reason += "; ";
        report->reason += message.str();
    }
    if (expected_ef_construction != 0 &&
        metadata.ef_construction != 0 &&
        metadata.ef_construction != expected_ef_construction) {
        std::ostringstream message;
        message << "index parameter conflict: file ef_construction "
                << metadata.ef_construction
                << " does not match requested ef_construction "
                << expected_ef_construction;
        report->parameter_conflict = true;
        report->requires_rebuild = true;
        report->loadable = false;
        report->status = "requires_rebuild";
        if (!report->reason.empty())
            report->reason += "; ";
        report->reason += message.str();
    }
}

inline IndexCompatibilityReport inspectIndexFile(
    const std::string &path, uint32_t expected_space_id = INDEX_SPACE_UNKNOWN,
    uint32_t expected_dim = 0, uint64_t expected_m = 0,
    uint64_t expected_ef_construction = 0,
    uint32_t reader_version = INDEX_READER_VERSION) {
    IndexCompatibilityReport report;
    report.path = path;

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        report.reason = "cannot open index file";
        return report;
    }
    input.seekg(0, std::ios::end);
    std::streampos end = input.tellg();
    input.seekg(0, std::ios::beg);
    if (!input || end < 0) {
        input.clear();
        input.seekg(0, std::ios::beg);
        IndexCompatibilityReport fallback;
        fallback.path = path;
        fallback.readable = true;
        fallback.reason = "cannot determine index file size";
        return fallback;
    }
    report.readable = true;
    const uint64_t file_size = static_cast<uint64_t>(end);
    if (file_size < legacyHeaderSize()) {
        report.truncated = true;
        report.reason = "index file is truncated: it is smaller than the "
                        "historical index header";
        return report;
    }

    IndexFormatMetadata footer_metadata;
    std::string footer_error;
    const bool has_footer =
        readCompatFooter(input, file_size, &footer_metadata, &footer_error);
    IndexFormatMetadata header_metadata;
    std::string header_error;
    bool has_header = false;
    if (has_footer) {
        has_header = readCompatHeaderAt(input, footer_metadata.header_offset,
                                        &header_metadata, &header_error);
    } else {
        has_header = readCompatHeaderAt(input, legacyHeaderSize(),
                                        &header_metadata, &header_error);
    }

    if (!has_footer && !has_header) {
        LegacyIndexHeader legacy_header;
        std::string legacy_error;
        if (!readLegacyHeader(input, &legacy_header, &legacy_error)) {
            report.truncated = true;
            report.reason = legacy_error;
            return report;
        }
        const uint64_t legacy_links_size =
            static_cast<uint64_t>(legacy_header.max_m0) * sizeof(uint32_t) +
            sizeof(uint32_t);
        if (legacy_header.offset_level0 != 0 ||
            legacy_header.max_elements < legacy_header.element_count ||
            legacy_header.graph_m == 0 ||
            legacy_header.max_m0 != legacy_header.graph_m * 2 ||
            legacy_header.size_data_per_element <=
                legacy_links_size + sizeof(size_t)) {
            report.reason =
                "file is not a recognized legacy hnswlib index header";
            return report;
        }
        const uint64_t legacy_data_size =
            legacy_header.size_data_per_element - legacy_links_size -
            sizeof(size_t);
        if (legacy_data_size % sizeof(float) != 0) {
            report.reason =
                "legacy index vector payload is not aligned to float data";
            return report;
        }
        report.legacy = true;
        report.metadata.present = false;
        report.metadata.format_version = INDEX_FORMAT_LEGACY;
        report.metadata.min_reader_version = INDEX_FORMAT_LEGACY;
        report.metadata.max_elements = legacy_header.max_elements;
        report.metadata.element_count = legacy_header.element_count;
        report.metadata.graph_m = legacy_header.graph_m;
        report.metadata.ef_construction = legacy_header.ef_construction;
        report.metadata.size_data_per_element =
            legacy_header.size_data_per_element;
        report.metadata.offset_data = legacy_header.offset_data;
        report.metadata.label_offset = legacy_header.label_offset;
        report.metadata.file_size = file_size;
        report.metadata.distance_type_size = sizeof(float);
        report.metadata.dim =
            static_cast<uint32_t>(legacy_data_size / sizeof(float));
        uint64_t minimum_size = 0;
        if (!legacyMinimumSize(legacy_header, &minimum_size) ||
            file_size < minimum_size) {
            report.truncated = true;
            report.reason = "index file is truncated: declared element "
                            "layout does not fit in the file";
            return report;
        }
        report.loadable = true;
        report.upgradeable = true;
        report.requires_rebuild = false;
        report.status = "upgradeable";
        report.reason = "legacy format v1 is loadable; re-save it to add "
                        "version and parameter metadata";
        applyParameterChecks(&report, expected_space_id, expected_dim,
                             expected_m, expected_ef_construction);
        return report;
    }

    if (!has_footer) {
        report.truncated = true;
        report.reason = "index compatibility footer is missing; the file is "
                        "truncated or was not written completely";
        return report;
    }
    if (!has_header) {
        report.truncated = true;
        report.reason = header_error.empty()
                            ? "index compatibility header is missing"
                            : header_error;
        return report;
    }
    if (!metadataEquals(footer_metadata, header_metadata)) {
        report.reason = "index compatibility header and footer disagree; "
                        "the file is corrupted";
        return report;
    }

    report.metadata = footer_metadata;
    report.legacy = false;
    if (footer_metadata.min_reader_version > reader_version) {
        std::ostringstream message;
        message << "index format v" << footer_metadata.format_version
                << " requires reader v" << footer_metadata.min_reader_version
                << ", current reader is v" << reader_version
                << "; refusing to load (upgrade the reader or rebuild the "
                   "index)";
        report.reason = message.str();
        return report;
    }
    if (footer_metadata.body_byte_order != hostByteOrder()) {
        report.endian_mismatch = true;
        std::ostringstream message;
        message << "index body byte order is "
                << byteOrderName(footer_metadata.body_byte_order)
                << " but this platform is "
                << byteOrderName(hostByteOrder())
                << "; byte swapping is not supported, rebuild the index on "
                   "the reader platform";
        report.reason = message.str();
        return report;
    }
    if (footer_metadata.size_t_size != sizeof(size_t) ||
        footer_metadata.pointer_size != sizeof(void *)) {
        report.layout_mismatch = true;
        std::ostringstream message;
        message << "index layout was written with size_t="
                << footer_metadata.size_t_size << ", pointer="
                << footer_metadata.pointer_size << " but reader uses size_t="
                << sizeof(size_t) << ", pointer=" << sizeof(void *)
                << "; rebuild the index for this platform";
        report.reason = message.str();
        return report;
    }

    uint64_t body_end = 0;
    uint64_t expected_size = 0;
    if (!checkedAdd(footer_metadata.header_offset,
                    footer_metadata.header_size, &body_end) ||
        !checkedAdd(body_end, footer_metadata.body_size, &body_end) ||
        !checkedAdd(body_end, INDEX_COMPAT_FOOTER_SIZE, &expected_size)) {
        report.reason = "index metadata contains overflowing size fields";
        return report;
    }
    if (footer_metadata.file_size != expected_size ||
        file_size != expected_size) {
        report.truncated = file_size < expected_size;
        std::ostringstream message;
        message << "index file "
                << (report.truncated ? "is truncated" : "has trailing bytes")
                << ": metadata declares " << expected_size
                << " bytes, found " << file_size;
        report.reason = message.str();
        return report;
    }

    addAlignmentWarning(footer_metadata, &report.warnings);
    report.loadable = true;
    report.requires_rebuild = false;
    report.upgradeable = false;
    report.status = "compatible";
    report.reason = "same-format index; header compatibility checks passed";
    applyParameterChecks(&report, expected_space_id, expected_dim,
                         expected_m, expected_ef_construction);
    return report;
}

inline IndexCompatibilityReport inspectIndexFile(
    const std::string &path, const std::string &expected_space,
    uint32_t expected_dim = 0, uint64_t expected_m = 0,
    uint64_t expected_ef_construction = 0,
    uint32_t reader_version = INDEX_READER_VERSION) {
    return inspectIndexFile(path, spaceIdFromName(expected_space),
                            expected_dim, expected_m,
                            expected_ef_construction, reader_version);
}

inline std::vector<IndexCompatibilityReport> inspectIndexFiles(
    const std::vector<std::string> &paths,
    uint32_t expected_space_id = INDEX_SPACE_UNKNOWN,
    uint32_t expected_dim = 0, uint64_t expected_m = 0,
    uint64_t expected_ef_construction = 0,
    uint32_t reader_version = INDEX_READER_VERSION) {
    std::vector<IndexCompatibilityReport> reports;
    reports.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        reports.push_back(inspectIndexFile(
            paths[i], expected_space_id, expected_dim, expected_m,
            expected_ef_construction, reader_version));
    }
    return reports;
}

}  // namespace hnswlib

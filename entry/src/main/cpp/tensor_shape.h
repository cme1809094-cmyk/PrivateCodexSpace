#ifndef PERMISSION_TENSOR_SHAPE_H
#define PERMISSION_TENSOR_SHAPE_H

#include "bridge_error.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace permission_nnrt {
inline std::string DescribeShape(const int32_t *shape, size_t rank)
{
    if (!shape) return "<null>";
    std::string text = "[";
    // Bound formatting of malformed or unexpectedly large descriptors.
    for (size_t i = 0; i < rank && i < 8; ++i) {
        if (i) text += ',';
        text += std::to_string(shape[i]);
    }
    if (rank > 8) text += ",...";
    return text + ']';
}

inline size_t TokenSequenceLength(const int32_t *shape, size_t rank)
{
    // A token tensor is one contiguous vector. Unit dimensions do not change
    // its order or byte count: [L], [1,L], [1,L,1,1], [1,1,1,L] are equivalent.
    size_t length = 1;
    bool valid = shape && rank > 0 && rank <= 8;
    if (valid) {
        for (size_t i = 0; i < rank; ++i) {
            if (shape[i] == 1) continue;
            if (length != 1 || (shape[i] != 16 && shape[i] != 32 && shape[i] != 64)) {
                valid = false;
                break;
            }
            length = static_cast<size_t>(shape[i]);
        }
    }
    if (!valid || length == 1) {
        throw BridgeError("Unsupported token input shape " + DescribeShape(shape, rank)
            + " (rank=" + std::to_string(rank)
            + "); expected one 16/32/64-token vector with optional unit dimensions");
    }
    return length;
}
} // namespace permission_nnrt
#endif

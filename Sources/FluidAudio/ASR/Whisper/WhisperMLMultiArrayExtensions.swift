import CoreML

// Adapted from ArgmaxCore/MLMultiArrayExtensions.swift in WhisperKit
// MIT License, Copyright © 2024 Argmax, Inc.

// MARK: - MLMultiArray Indexing & Fill

extension MLMultiArray {
    /// Computes the linear offset from multi-dimensional indices using strides.
    @inline(__always)
    func linearOffset(for index: [NSNumber], strides strideInts: [Int]? = nil) -> Int {
        var linearOffset = 0
        let strideInts = strideInts ?? strides.map { $0.intValue }
        for (dimension, stride) in zip(index, strideInts) {
            linearOffset += dimension.intValue * stride
        }
        return linearOffset
    }

    /// Fills a contiguous range of indices in the last dimension with a value.
    /// Array must have shape [1, 1, n].
    func fillLastDimension(indexes: Range<Int>, with value: Float16) {
        precondition(shape.count == 3 && shape[0] == 1 && shape[1] == 1, "Must have [1, 1, n] shape")
        withUnsafeMutableBytes { ptr, strides in
            let base = ptr.baseAddress!.bindMemory(to: Float16.self, capacity: count)
            for index in indexes {
                base[index * strides[2]] = value
            }
        }
    }

    /// Fills specific multi-dimensional indices with a Float16 value.
    func fill(indexes: [[NSNumber]], with value: Float16) {
        let pointer = dataPointer.bindMemory(to: Float16.self, capacity: count)
        let strideInts = strides.map { $0.intValue }
        for index in indexes {
            pointer[linearOffset(for: index, strides: strideInts)] = value
        }
    }

    /// Fills specific multi-dimensional indices with a Float value.
    func fill(indexes: [[NSNumber]], with value: Float) {
        let pointer = dataPointer.bindMemory(to: Float.self, capacity: count)
        let strideInts = strides.map { $0.intValue }
        for index in indexes {
            pointer[linearOffset(for: index, strides: strideInts)] = value
        }
    }
}

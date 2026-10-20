// SPDX-License-Identifier: MIT

import Cmlx

/// A strongly-typed identity of an ``MLXArray``'s UNDERLYING STORAGE, sampled from
/// the native MLX runtime without forcing evaluation.
///
/// This is deliberately NOT:
///   * `ObjectIdentifier(array)` — the Swift wrapper's identity. Two distinct
///     `MLXArray` objects (or views) can share one backing buffer, and comparing
///     the wrapper proves nothing about that aliasing.
///   * `mlx::core::array::id()` — which names the per-array `ArrayDesc`, not the
///     shared data, so different ids do not rule out a shared allocation.
///
/// It carries the two pieces of evidence MLX actually owns, exposed by
/// ``MLXArray/mlxStorageIdentity``: the shared `Data` descriptor address and the
/// allocator buffer handle. Aliasing holds if EITHER matches; a buffer whose
/// identity cannot be confirmed (an unmaterialized array) is `.indeterminate` and
/// must fail closed for any mutable-isolation claim — the two evidences are not
/// interchangeable, and a missing one is never grounds to declare arrays distinct.
public struct MLXArrayStorageIdentity: Hashable, Sendable {
    /// Address of the shared `mlx::core::array::Data` (0 when unmaterialized).
    public let dataDescriptor: UInt
    /// Allocator buffer base handle (0 when unmaterialized).
    public let bufferHandle: UInt
    /// True only when the array already owns materialized data; false means the
    /// underlying storage identity is NOT observable.
    public let materialized: Bool

    /// Public initializer so downstream targets (and native tests) can construct
    /// an identity value directly from the two evidence axes.
    public init(dataDescriptor: UInt, bufferHandle: UInt, materialized: Bool) {
        self.dataDescriptor = dataDescriptor
        self.bufferHandle = bufferHandle
        self.materialized = materialized
    }

    /// The non-zero storage evidences this array exposes.
    private var evidences: Set<UInt> {
        var s: Set<UInt> = []
        if dataDescriptor != 0 { s.insert(dataDescriptor) }
        if bufferHandle != 0 { s.insert(bufferHandle) }
        return s
    }

    /// The relationship between two arrays' underlying storage.
    public enum Relationship: Hashable, Sendable {
        /// They alias the same allocation (shared `Data` descriptor or buffer).
        case shared
        /// Both are materialized and provably own disjoint storage.
        case distinct
        /// At least one side's storage identity cannot be confirmed; a mutable
        /// isolation claim over it must fail closed.
        case indeterminate
    }

    /// Classify sharing against another identity WITHOUT trusting a single axis.
    /// Any matching evidence (descriptor or buffer) means shared; if either side
    /// is unmaterialized the relationship is indeterminate — we refuse to declare
    /// two arrays distinct just because one piece of evidence did not collide.
    public func relationship(to other: MLXArrayStorageIdentity) -> Relationship {
        if !materialized || !other.materialized {
            // Missing allocation evidence: we cannot certify non-sharing. Report
            // `shared` first if any evidence still collides, else indeterminate.
            if !evidences.isDisjoint(with: other.evidences) { return .shared }
            return .indeterminate
        }
        return evidences.isDisjoint(with: other.evidences) ? .distinct : .shared
    }
}

public extension MLXArray {
    /// The underlying storage identity of this array, read from the native runtime.
    /// Never evaluates the array; an unmaterialized array reports
    /// `materialized == false` so callers fail closed instead of assuming distinct.
    var mlxStorageIdentity: MLXArrayStorageIdentity {
        let raw = mlx_array_get_storage_identity(ctx)
        return MLXArrayStorageIdentity(
            dataDescriptor: UInt(raw.data_descriptor),
            bufferHandle: UInt(raw.buffer),
            materialized: raw.holds_data)
    }
}

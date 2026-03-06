/// A serializing work queue that allows at most one concurrent async operation,
/// with a bounded FIFO waiting list of suspended callers.
///
/// ```swift
/// let queue = SerialWorkQueue(maxDepth: 4)
///
/// try await queue.acquire()
/// defer { Task { await queue.release() } }
/// // … do exclusive work …
/// ```
///
/// Callers that arrive while the slot is occupied are suspended until the current
/// holder calls `release()`. Callers beyond `maxDepth` receive a thrown
/// `SerialWorkQueueError.queueFull` immediately rather than suspending.
public actor SerialWorkQueue {

    /// Maximum number of callers that may be suspended waiting for the slot.
    public let maxDepth: Int

    private var occupied = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(maxDepth: Int = 4) {
        self.maxDepth = maxDepth
    }

    // MARK: - Observable State

    /// The number of callers currently suspended waiting for the slot.
    public var pendingCount: Int { waiters.count }

    /// Whether the slot is currently occupied.
    public var isOccupied: Bool { occupied }

    // MARK: - Acquire / Release

    /// Acquire the slot. Returns immediately if free; suspends the caller if busy.
    ///
    /// When this returns without throwing, the caller owns the slot and **must**
    /// eventually call `release()` — including on error paths.
    ///
    /// - Throws: `SerialWorkQueueError.queueFull` when `maxDepth` callers are
    ///   already suspended and the slot is still occupied.
    public func acquire() async throws {
        guard occupied else {
            occupied = true
            return
        }
        guard waiters.count < maxDepth else {
            throw SerialWorkQueueError.queueFull(maxDepth: maxDepth)
        }
        // Suspend; `release()` will resume us when the slot is free.
        await withCheckedContinuation { waiters.append($0) }
        // When we wake, `occupied` is still true — ownership has been transferred to us.
    }

    /// Release the slot. Resumes the earliest suspended caller (FIFO) and transfers
    /// ownership to them; if no callers are waiting, marks the slot free.
    public func release() {
        guard let next = waiters.first else {
            occupied = false
            return
        }
        waiters.removeFirst()
        next.resume()  // `occupied` stays true — next caller is now the owner
    }
}

// MARK: - Error

/// Errors thrown by `SerialWorkQueue.acquire()`.
public enum SerialWorkQueueError: Error, Sendable {
    /// The waiter queue is full; the request was not enqueued.
    case queueFull(maxDepth: Int)
}

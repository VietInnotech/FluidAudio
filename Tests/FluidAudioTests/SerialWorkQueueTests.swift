import XCTest

@testable import FluidAudio

final class SerialWorkQueueTests: XCTestCase {

    // MARK: - Basic Acquire / Release

    func testAcquire_whenFree_succeedsImmediatelyAndMarksOccupied() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        var occupied = await queue.isOccupied
        var pending = await queue.pendingCount
        XCTAssertFalse(occupied)
        XCTAssertEqual(pending, 0)

        try await queue.acquire()

        occupied = await queue.isOccupied
        pending = await queue.pendingCount
        XCTAssertTrue(occupied)
        XCTAssertEqual(pending, 0)

        await queue.release()
    }

    func testRelease_afterAcquire_marksFree() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        try await queue.acquire()
        var occupied = await queue.isOccupied
        XCTAssertTrue(occupied)

        await queue.release()

        occupied = await queue.isOccupied
        let pending = await queue.pendingCount
        XCTAssertFalse(occupied)
        XCTAssertEqual(pending, 0)
    }

    func testRelease_whenAlreadyFree_isNoOp() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        // Releasing an un-acquired queue must not crash or corrupt state.
        await queue.release()

        let occupied = await queue.isOccupied
        let pending = await queue.pendingCount
        XCTAssertFalse(occupied)
        XCTAssertEqual(pending, 0)
    }

    // MARK: - Blocking / Unblocking

    func testSecondCaller_suspends_untilRelease() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        // Occupy the slot.
        try await queue.acquire()

        // Spawn a concurrent caller that must wait.
        let waiterTask = Task<Bool, any Error> {
            try await queue.acquire()
            let wasOccupied = await queue.isOccupied
            await queue.release()
            return wasOccupied
        }

        // Give the waiter enough time to reach acquire() and suspend.
        try await waitForPendingCount(queue, expected: 1)
        let pending = await queue.pendingCount
        let occupied = await queue.isOccupied
        XCTAssertEqual(pending, 1)
        XCTAssertTrue(occupied)

        // Release — unblocks the waiter.
        await queue.release()

        // Waiter should complete and have observed isOccupied == true while holding the slot.
        let waiterSawOccupied = try await waiterTask.value
        XCTAssertTrue(waiterSawOccupied, "waiter should observe isOccupied == true while it holds the slot")

        let finalOccupied = await queue.isOccupied
        let finalPending = await queue.pendingCount
        XCTAssertFalse(finalOccupied)
        XCTAssertEqual(finalPending, 0)
    }

    func testRelease_withWaiters_keepsBusyAndTransfersOwnership() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        try await queue.acquire()

        // Two waiters.
        let w1 = Task<Void, any Error> { try await queue.acquire(); await queue.release() }
        let w2 = Task<Void, any Error> { try await queue.acquire(); await queue.release() }

        try await waitForPendingCount(queue, expected: 2)

        // Release: next waiter takes over, isOccupied stays true.
        await queue.release()
        let stillOccupied = await queue.isOccupied
        XCTAssertTrue(stillOccupied, "slot should still be occupied by first waiter after release")

        try await w1.value
        try await w2.value

        let finalOccupied = await queue.isOccupied
        XCTAssertFalse(finalOccupied)
    }

    // MARK: - Queue Full Rejection

    func testQueueFull_rejectsOnceMaxDepthReached() async throws {
        let maxDepth = 2
        let queue = SerialWorkQueue(maxDepth: maxDepth)

        // Occupy the slot.
        try await queue.acquire()

        // Fill the waiter queue up to maxDepth.
        var waiterTasks: [Task<Void, any Error>] = []
        for _ in 0..<maxDepth {
            let t = Task<Void, any Error> { try await queue.acquire(); await queue.release() }
            waiterTasks.append(t)
        }

        try await waitForPendingCount(queue, expected: maxDepth)
        let pendingAtFull = await queue.pendingCount
        XCTAssertEqual(pendingAtFull, maxDepth)

        // The next caller should be rejected immediately.
        do {
            try await queue.acquire()
            XCTFail("Expected SerialWorkQueueError.queueFull but no error was thrown")
        } catch SerialWorkQueueError.queueFull(let d) {
            XCTAssertEqual(d, maxDepth, "reported maxDepth in error should match configured maxDepth")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        // Drain.
        await queue.release()
        for t in waiterTasks { try await t.value }

        let finalOccupied = await queue.isOccupied
        XCTAssertFalse(finalOccupied)
    }

    func testQueueFull_doesNotAffectExistingWaiters() async throws {
        let maxDepth = 1
        let queue = SerialWorkQueue(maxDepth: maxDepth)

        try await queue.acquire()

        // One valid waiter.
        let validWaiter = Task<Void, any Error> { try await queue.acquire(); await queue.release() }
        try await waitForPendingCount(queue, expected: 1)

        // Rejected caller.
        do {
            try await queue.acquire()
        } catch SerialWorkQueueError.queueFull {}

        // Drain — valid waiter should still complete.
        await queue.release()
        try await validWaiter.value

        let finalOccupied = await queue.isOccupied
        let finalPending = await queue.pendingCount
        XCTAssertFalse(finalOccupied)
        XCTAssertEqual(finalPending, 0)
    }

    // MARK: - FIFO Ordering

    func testFIFOOrdering_waitersAreReleasedInOrder() async throws {
        let queue = SerialWorkQueue(maxDepth: 4)

        actor OrderTracker {
            private(set) var completions: [Int] = []
            func record(_ n: Int) { completions.append(n) }
        }
        let tracker = OrderTracker()

        // Hold the slot so all subsequent callers queue up.
        try await queue.acquire()

        // Enqueue 4 waiters in order. We wait after each until pendingCount grows
        // to confirm the task has actually reached acquire() before spawning the next.
        var tasks: [Task<Void, any Error>] = []
        for i in 1...4 {
            let position = i
            let t = Task<Void, any Error> {
                try await queue.acquire()
                await tracker.record(position)
                await queue.release()
            }
            tasks.append(t)
            // Wait until this task is visible in the waiter list.
            try await waitForPendingCount(queue, expected: i)
        }

        let enqueued = await queue.pendingCount
        XCTAssertEqual(enqueued, 4)

        // Release — the four waiters should drain in FIFO order.
        await queue.release()
        for t in tasks { try await t.value }

        let order = await tracker.completions
        XCTAssertEqual(order, [1, 2, 3, 4], "waiters must be released in FIFO order")
    }

    // MARK: - Multiple Acquire/Release Cycles

    func testMultipleCycles_slotRemainsHealthy() async throws {
        let queue = SerialWorkQueue(maxDepth: 3)

        for cycle in 1...10 {
            let before = await queue.isOccupied
            XCTAssertFalse(before, "cycle \(cycle): slot should be free before acquire")
            try await queue.acquire()
            let during = await queue.isOccupied
            XCTAssertTrue(during, "cycle \(cycle): slot should be occupied after acquire")
            await queue.release()
        }

        let finalOccupied = await queue.isOccupied
        let finalPending = await queue.pendingCount
        XCTAssertFalse(finalOccupied)
        XCTAssertEqual(finalPending, 0)
    }

    // MARK: - Helpers

    /// Polls `queue.pendingCount` until it reaches `expected`, timing out after ~1 s.
    private func waitForPendingCount(
        _ queue: SerialWorkQueue,
        expected: Int,
        file: StaticString = #file,
        line: UInt = #line
    ) async throws {
        let pollInterval: UInt64 = 10_000_000  // 10 ms
        let maxPolls = 100  // 1 s total

        for _ in 0..<maxPolls {
            if await queue.pendingCount >= expected { return }
            try await Task.sleep(nanoseconds: pollInterval)
        }

        let actual = await queue.pendingCount
        XCTFail(
            "Timed out waiting for pendingCount == \(expected); got \(actual)",
            file: file,
            line: line
        )
    }
}

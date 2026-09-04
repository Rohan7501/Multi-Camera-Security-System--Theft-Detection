// Unit tests for FrameQueue<T> -- the bounded queue between the gRPC handler
// threads and the inference worker pool.
//
// Its drop-oldest-on-overflow policy is what keeps a slow detector from applying
// backpressure to camera capture, so the drop accounting and the shutdown wakeup
// are both load-bearing. No gRPC, no CUDA, no camera: this runs anywhere.

#include "frame_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static void test_push_pop_roundtrip()
{
    FrameQueue<int> q(4);
    q.push(7);
    int got = 0;
    CHECK(q.pop(got));
    CHECK(got == 7);
    CHECK(q.size() == 0);
    CHECK(q.dropped() == 0);
}

static void test_fifo_order()
{
    FrameQueue<int> q(8);
    for (int i = 0; i < 5; ++i) q.push(i);
    for (int i = 0; i < 5; ++i) {
        int got = -1;
        CHECK(q.pop(got));
        CHECK(got == i);
    }
}

static void test_drops_oldest_when_full()
{
    // Capacity 3, five pushes: 0 and 1 are evicted, 2/3/4 survive. Dropping the
    // OLDEST is the point -- a live pipeline wants the newest frames.
    FrameQueue<int> q(3);
    for (int i = 0; i < 5; ++i) q.push(i);

    CHECK(q.size() == 3);
    CHECK(q.dropped() == 2);

    for (int expect : {2, 3, 4}) {
        int got = -1;
        CHECK(q.pop(got));
        CHECK(got == expect);
    }
}

static void test_dropped_counter_is_cumulative()
{
    // The metrics sampler mirrors this counter by delta, so it must never reset.
    FrameQueue<int> q(1);
    for (int i = 0; i < 10; ++i) q.push(i);
    CHECK(q.dropped() == 9);

    int got = -1;
    q.pop(got);
    CHECK(q.dropped() == 9);      // draining must not clear the count
}

static void test_shutdown_wakes_a_blocked_pop()
{
    // Every worker thread blocks in pop(); shutdown() is the only thing that
    // releases them, and main.cpp's join loop hangs forever if it doesn't.
    FrameQueue<int> q(4);
    std::atomic<bool> returned{false};
    std::atomic<bool> result{true};

    std::thread t([&] {
        int got = 0;
        result = q.pop(got);       // blocks: queue is empty
        returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!returned);              // still blocked, as expected

    q.shutdown();
    t.join();

    CHECK(returned);
    CHECK(!result);                // pop returns false on shutdown
}

static void test_shutdown_releases_every_waiter()
{
    // notify_all, not notify_one: with NUM_THREADS workers parked on one queue,
    // waking a single waiter would leave the rest blocked and the process would
    // never exit.
    FrameQueue<int> q(4);
    constexpr int kWaiters = 6;
    std::atomic<int> woke{0};
    std::vector<std::thread> ts;

    for (int i = 0; i < kWaiters; ++i)
        ts.emplace_back([&] {
            int got = 0;
            q.pop(got);
            ++woke;
        });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.shutdown();
    for (auto& t : ts) t.join();

    CHECK(woke == kWaiters);
}

static void test_concurrent_producers_lose_nothing()
{
    // pushed == popped + dropped is the invariant that says the queue never
    // silently loses an item under contention.
    constexpr int kProducers = 4, kPerProducer = 500;
    FrameQueue<int> q(64);
    std::atomic<int> popped{0};

    std::thread consumer([&] {
        int got = 0;
        while (q.pop(got)) ++popped;
    });

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p)
        producers.emplace_back([&] {
            for (int i = 0; i < kPerProducer; ++i) q.push(i);
        });
    for (auto& t : producers) t.join();

    // Let the consumer drain, then release it.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const uint64_t dropped = q.dropped();
    const size_t remaining = q.size();
    q.shutdown();
    consumer.join();

    const int pushed = kProducers * kPerProducer;
    CHECK(static_cast<uint64_t>(pushed) ==
          static_cast<uint64_t>(popped) + dropped + remaining);
}

int main()
{
    test_push_pop_roundtrip();
    test_fifo_order();
    test_drops_oldest_when_full();
    test_dropped_counter_is_cumulative();
    test_shutdown_wakes_a_blocked_pop();
    test_shutdown_releases_every_waiter();
    test_concurrent_producers_lose_nothing();

    if (failures == 0) {
        std::printf("test_frame_queue: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_frame_queue: %d check(s) failed\n", failures);
    return 1;
}

// #pragma once
// #include <queue>
// #include <mutex>
// #include <condition_variable>

// template<typename T>
// class FrameQueue
// {
// public:

//     void push(const T& item)
//     {
//         std::unique_lock<std::mutex> lock(mutex_);
//         queue_.push(item);
//         cv_.notify_one();
//     }

//     T pop()
//     {
//         std::unique_lock<std::mutex> lock(mutex_);

//         cv_.wait(lock, [&]{ return !queue_.empty(); });

//         T item = queue_.front();
//         queue_.pop();

//         return item;
//     }

// private:

//     std::queue<T> queue_;
//     std::mutex mutex_;
//     std::condition_variable cv_;
// };
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstdint>

template<typename T>
class FrameQueue
{
public:

    FrameQueue(size_t max_size)
        : max_size_(max_size) {}

    void push(const T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if(queue_.size() >= max_size_)
        {
            // drop oldest frame
            queue_.pop();
            dropped_++;
        }

        queue_.push(item);

        cv_.notify_one();
    }

    // Observability (exported as inference_queue_depth / _dropped_total).
    size_t size() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    uint64_t dropped() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return dropped_;
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [&]{
            return !queue_.empty() || shutdown_;
        });

        if(shutdown_)
            return false;

        item = queue_.front();
        queue_.pop();

        return true;
    }

    void shutdown()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        shutdown_ = true;

        cv_.notify_all();
    }

private:

    std::queue<T> queue_;
    size_t max_size_;

    mutable std::mutex mutex_;      // mutable: size()/dropped() are const
    std::condition_variable cv_;

    uint64_t dropped_ = 0;
    bool shutdown_ = false;
};

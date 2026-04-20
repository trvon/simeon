#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace simeon {

// Open-addressing hash map specialized for uint64 keys. Used by Bm25Index
// for the insert-heavy postings maps where std::unordered_map dominated
// build time (~27% of total wall time in the SAB-smooth profile audit).
//
// Design choices:
//   - Identity hash. Keys come from hash64() which produces high-entropy
//     uint64_t values; re-hashing would waste cycles.
//   - Linear probing with power-of-two capacity so bucket index is
//     key & mask_.
//   - Parallel occupancy vector (uint8_t per slot) to distinguish empty
//     from "occupied with default V" without reserving a key sentinel.
//   - Insert-only API. No erase(). Bm25Index never removes postings.
//   - Stored value_type is std::pair<std::uint64_t, V> (AoS) so iterators
//     can return real references and structured bindings work the same
//     way they did against std::unordered_map.
template <class V> class FlatHashMapU64 {
    static constexpr std::size_t kDefaultCapacity = 16;
    static constexpr double kMaxLoadFactor = 0.70;

public:
    using key_type = std::uint64_t;
    using mapped_type = V;
    using value_type = std::pair<std::uint64_t, V>;

    FlatHashMapU64() = default;

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t capacity() const noexcept { return slots_.size(); }

    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<std::uint64_t, V>;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        iterator() = default;
        iterator(FlatHashMapU64* m, std::size_t idx) noexcept : m_(m), idx_(idx) { advance(); }

        reference operator*() const noexcept { return m_->slots_[idx_]; }
        pointer operator->() const noexcept { return &m_->slots_[idx_]; }
        iterator& operator++() noexcept {
            ++idx_;
            advance();
            return *this;
        }
        bool operator==(const iterator& o) const noexcept { return m_ == o.m_ && idx_ == o.idx_; }
        bool operator!=(const iterator& o) const noexcept { return !(*this == o); }

    private:
        void advance() noexcept {
            const std::size_t cap = m_ ? m_->capacity() : 0;
            while (idx_ < cap && !m_->occupied_[idx_])
                ++idx_;
        }
        FlatHashMapU64* m_ = nullptr;
        std::size_t idx_ = 0;
    };

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<std::uint64_t, V>;
        using reference = const value_type&;
        using pointer = const value_type*;
        using difference_type = std::ptrdiff_t;

        const_iterator() = default;
        const_iterator(const FlatHashMapU64* m, std::size_t idx) noexcept : m_(m), idx_(idx) {
            advance();
        }

        reference operator*() const noexcept { return m_->slots_[idx_]; }
        pointer operator->() const noexcept { return &m_->slots_[idx_]; }
        const_iterator& operator++() noexcept {
            ++idx_;
            advance();
            return *this;
        }
        bool operator==(const const_iterator& o) const noexcept {
            return m_ == o.m_ && idx_ == o.idx_;
        }
        bool operator!=(const const_iterator& o) const noexcept { return !(*this == o); }

    private:
        void advance() noexcept {
            const std::size_t cap = m_ ? m_->capacity() : 0;
            while (idx_ < cap && !m_->occupied_[idx_])
                ++idx_;
        }
        const FlatHashMapU64* m_ = nullptr;
        std::size_t idx_ = 0;
    };

    iterator begin() noexcept { return iterator(this, 0); }
    iterator end() noexcept { return iterator(this, capacity()); }
    const_iterator begin() const noexcept { return const_iterator(this, 0); }
    const_iterator end() const noexcept { return const_iterator(this, capacity()); }

    iterator find(std::uint64_t key) noexcept {
        if (capacity() == 0)
            return end();
        std::size_t idx = key & mask_;
        while (occupied_[idx]) {
            if (slots_[idx].first == key)
                return iterator(this, idx);
            idx = (idx + 1) & mask_;
        }
        return end();
    }

    const_iterator find(std::uint64_t key) const noexcept {
        if (capacity() == 0)
            return end();
        std::size_t idx = key & mask_;
        while (occupied_[idx]) {
            if (slots_[idx].first == key)
                return const_iterator(this, idx);
            idx = (idx + 1) & mask_;
        }
        return end();
    }

    V& operator[](std::uint64_t key) {
        if (capacity() == 0)
            rehash(kDefaultCapacity);
        else if (static_cast<double>(size_ + 1) > kMaxLoadFactor * static_cast<double>(capacity()))
            rehash(capacity() * 2);

        std::size_t idx = key & mask_;
        while (occupied_[idx]) {
            if (slots_[idx].first == key)
                return slots_[idx].second;
            idx = (idx + 1) & mask_;
        }
        occupied_[idx] = 1;
        slots_[idx].first = key;
        slots_[idx].second = V{};
        ++size_;
        return slots_[idx].second;
    }

    void reserve(std::size_t n) {
        std::size_t target = 1;
        while (static_cast<double>(n) > kMaxLoadFactor * static_cast<double>(target))
            target <<= 1;
        if (target > capacity())
            rehash(target);
    }

private:
    void rehash(std::size_t new_cap) {
        // new_cap must be a power of two.
        std::vector<value_type> old_slots = std::move(slots_);
        std::vector<std::uint8_t> old_occ = std::move(occupied_);
        slots_ = std::vector<value_type>(new_cap);
        occupied_.assign(new_cap, 0);
        mask_ = new_cap - 1;
        size_ = 0;
        for (std::size_t i = 0; i < old_slots.size(); ++i) {
            if (!old_occ[i])
                continue;
            std::uint64_t k = old_slots[i].first;
            std::size_t idx = k & mask_;
            while (occupied_[idx])
                idx = (idx + 1) & mask_;
            occupied_[idx] = 1;
            slots_[idx].first = k;
            slots_[idx].second = std::move(old_slots[i].second);
            ++size_;
        }
    }

    std::vector<value_type> slots_;
    std::vector<std::uint8_t> occupied_;
    std::size_t size_ = 0;
    std::size_t mask_ = 0;
};

} // namespace simeon

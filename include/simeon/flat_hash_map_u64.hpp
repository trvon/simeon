#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace simeon {

// Open-addressing hash map specialized for uint64 keys. Used by Bm25Index
// for the insert-heavy postings maps where std::unordered_map dominated
// build time and memory.
//
// Design choices:
//   - Identity hash. Keys come from hash64() which produces high-entropy
//     uint64_t values; re-hashing would waste cycles.
//   - Linear probing with power-of-two capacity so bucket index is key & mask_.
//   - Parallel key + entry-index arrays for the slot directory; an entry index
//     of kEmptyEntry denotes an empty slot, so no separate occupancy bitmap is
//     needed.
//   - Dense entries array that stores actual (key, value) pairs only for
//     occupied buckets. This avoids paying for a full V in every empty slot.
//   - Insert-only API. No erase(). Bm25Index never removes postings.
template <class V> class FlatHashMapU64 {
    static constexpr std::size_t kDefaultCapacity = 16;
    static constexpr double kMaxLoadFactor = 0.70;
    static constexpr std::uint32_t kEmptyEntry = std::numeric_limits<std::uint32_t>::max();

public:
    using key_type = std::uint64_t;
    using mapped_type = V;
    using value_type = std::pair<std::uint64_t, V>;

    FlatHashMapU64() = default;

    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    std::size_t capacity() const noexcept { return keys_.size(); }

    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<std::uint64_t, V>;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        iterator() = default;
        iterator(FlatHashMapU64* m, std::size_t idx) noexcept : m_(m), idx_(idx) { advance(); }

        reference operator*() const noexcept { return m_->entries_[m_->entry_indices_[idx_]]; }
        pointer operator->() const noexcept { return &m_->entries_[m_->entry_indices_[idx_]]; }
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
            while (idx_ < cap && m_->entry_indices_[idx_] == kEmptyEntry)
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

        reference operator*() const noexcept { return m_->entries_[m_->entry_indices_[idx_]]; }
        pointer operator->() const noexcept { return &m_->entries_[m_->entry_indices_[idx_]]; }
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
            while (idx_ < cap && m_->entry_indices_[idx_] == kEmptyEntry)
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
        while (entry_indices_[idx] != kEmptyEntry) {
            if (keys_[idx] == key)
                return iterator(this, idx);
            idx = (idx + 1) & mask_;
        }
        return end();
    }

    const_iterator find(std::uint64_t key) const noexcept {
        if (capacity() == 0)
            return end();
        std::size_t idx = key & mask_;
        while (entry_indices_[idx] != kEmptyEntry) {
            if (keys_[idx] == key)
                return const_iterator(this, idx);
            idx = (idx + 1) & mask_;
        }
        return end();
    }

    V& operator[](std::uint64_t key) {
        if (capacity() == 0)
            rehash(kDefaultCapacity);
        else if (static_cast<double>(size() + 1) > kMaxLoadFactor * static_cast<double>(capacity()))
            rehash(capacity() * 2);

        std::size_t idx = key & mask_;
        while (entry_indices_[idx] != kEmptyEntry) {
            if (keys_[idx] == key)
                return entries_[entry_indices_[idx]].second;
            idx = (idx + 1) & mask_;
        }

        keys_[idx] = key;
        entry_indices_[idx] = static_cast<std::uint32_t>(entries_.size());
        entries_.emplace_back(key, V{});
        return entries_.back().second;
    }

    void reserve(std::size_t n) {
        std::size_t target = 1;
        while (static_cast<double>(n) > kMaxLoadFactor * static_cast<double>(target))
            target <<= 1;
        entries_.reserve(n);
        if (target > capacity())
            rehash(target);
    }

    // Reset all entries while keeping the allocated capacity. Useful for
    // thread-local reuse across iterations (e.g. per-doc TF accumulators
    // in BM25 indexing). Resets entry_indices_ to empty markers in O(capacity).
    void clear() noexcept {
        if (entries_.empty())
            return;
        std::fill(entry_indices_.begin(), entry_indices_.end(), kEmptyEntry);
        entries_.clear();
    }

    void shrink_to_fit() {
        if (entries_.empty()) {
            entries_.clear();
            entries_.shrink_to_fit();
            keys_.clear();
            keys_.shrink_to_fit();
            entry_indices_.clear();
            entry_indices_.shrink_to_fit();
            mask_ = 0;
            return;
        }

        std::size_t target = kDefaultCapacity;
        while (static_cast<double>(entries_.size()) >
               kMaxLoadFactor * static_cast<double>(target)) {
            target <<= 1;
        }

        entries_.shrink_to_fit();
        if (target != capacity()) {
            rehash(target);
        }
        keys_.shrink_to_fit();
        entry_indices_.shrink_to_fit();
    }

private:
    void rehash(std::size_t new_cap) {
        keys_.assign(new_cap, 0);
        entry_indices_.assign(new_cap, kEmptyEntry);
        mask_ = new_cap - 1;

        for (std::uint32_t entry_idx = 0; entry_idx < static_cast<std::uint32_t>(entries_.size());
             ++entry_idx) {
            const auto& entry = entries_[entry_idx];
            const std::uint64_t key = entry.first;
            std::size_t idx = key & mask_;
            while (entry_indices_[idx] != kEmptyEntry)
                idx = (idx + 1) & mask_;
            keys_[idx] = key;
            entry_indices_[idx] = entry_idx;
        }
    }

    std::vector<value_type> entries_;
    std::vector<key_type> keys_;
    std::vector<std::uint32_t> entry_indices_;
    std::size_t mask_ = 0;
};

} // namespace simeon

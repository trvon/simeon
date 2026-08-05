// SPDX-License-Identifier: GPL-3.0-or-later
// Deterministic concept-mining construction benchmark.
//
// Usage: simeon_concept_mining_bench [doc-count] [vocabulary-size] [output-jsonl]
// The default target measures uninstrumented latency. Separately compiled RSS
// and allocation targets isolate those probes from timing. Every target emits
// a complete ConceptIndex fingerprint.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "simeon/bm25.hpp"
#include "simeon/concept_mining.hpp"
#include "simeon/detail/concept_index_fingerprint.hpp"

#ifndef SIMEON_TRACK_ALLOCATIONS
#define SIMEON_TRACK_ALLOCATIONS 0
#endif
#ifndef SIMEON_SAMPLE_RSS
#define SIMEON_SAMPLE_RSS 0
#endif

namespace {

#if SIMEON_TRACK_ALLOCATIONS
thread_local bool gTrackAllocations = false;
thread_local std::uint64_t gAllocationCount = 0;
thread_local std::uint64_t gAllocationBytes = 0;

void recordAllocation(std::size_t size) noexcept {
    if (gTrackAllocations) {
        ++gAllocationCount;
        gAllocationBytes += size;
    }
}
#endif

std::size_t parseSize(const char* text, std::size_t fallback, std::size_t minimum,
                      std::size_t maximum) {
    if (text == nullptr)
        return fallback;
    try {
        const auto value = std::stoull(text);
        return std::clamp<std::size_t>(value, minimum, maximum);
    } catch (...) {
        return fallback;
    }
}

#if SIMEON_SAMPLE_RSS
std::uint64_t currentRssBytes() noexcept {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                     &count) == KERN_SUCCESS
               ? static_cast<std::uint64_t>(info.resident_size)
               : 0;
#elif defined(__linux__)
    std::ifstream statm{"/proc/self/statm"};
    std::uint64_t totalPages = 0;
    std::uint64_t residentPages = 0;
    if (!(statm >> totalPages >> residentPages))
        return 0;
    const auto pageSize = sysconf(_SC_PAGESIZE);
    return pageSize > 0 ? residentPages * static_cast<std::uint64_t>(pageSize) : 0;
#else
#error "SIMEON_SAMPLE_RSS requires Darwin or Linux"
#endif
}
#endif

std::vector<std::string> makeCorpus(std::size_t docCount, std::size_t vocabularySize) {
    std::vector<std::string> docs;
    docs.reserve(docCount);
    for (std::size_t doc = 0; doc < docCount; ++doc) {
        const auto topic = doc % 64;
        std::string text;
        text.reserve(1400);
        for (std::size_t repetition = 0; repetition < 12; ++repetition) {
            text += "topic" + std::to_string(topic) + " concept" + std::to_string(topic) + ' ';
            for (std::size_t word = 0; word < 8; ++word) {
                const auto token = (doc * 131 + repetition * 17 + word * 7) % vocabularySize;
                text += "word" + std::to_string(token) + ' ';
            }
        }
        docs.push_back(std::move(text));
    }
    return docs;
}

bool writeRecord(const std::string& record, int argc, char** argv) {
    std::cout << record << '\n';
    if (argc <= 3)
        return true;
    std::ofstream output{argv[3], std::ios::app};
    if (!output) {
        std::cerr << "failed to open output: " << argv[3] << '\n';
        return false;
    }
    output << record << '\n';
    return true;
}

} // namespace

#if SIMEON_TRACK_ALLOCATIONS
void* operator new(std::size_t size) {
    if (void* memory = std::malloc(std::max<std::size_t>(size, 1))) {
        recordAllocation(size);
        return memory;
    }
    throw std::bad_alloc{};
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}
#endif

int main(int argc, char** argv) {
    const auto docCount = parseSize(argc > 1 ? argv[1] : nullptr, 10'000, 100, 100'000);
    const auto vocabularySize = parseSize(argc > 2 ? argv[2] : nullptr, 4096, 64, 100'000);

    auto docs = makeCorpus(docCount, vocabularySize);
    std::vector<std::string_view> views;
    views.reserve(docs.size());
    simeon::Bm25Index index;
    index.reserve_docs(docs.size());
    for (const auto& doc : docs) {
        views.emplace_back(doc);
        index.add_doc(doc);
    }
    index.finalize();

    simeon::ConceptConfig config;
    config.min_ttf = 5;
    config.pmi_floor = 1.0F;
    config.max_concepts = 200'000;

#if SIMEON_TRACK_ALLOCATIONS
    gAllocationCount = 0;
    gAllocationBytes = 0;
    gTrackAllocations = true;
    auto concepts = simeon::mine_concepts(index, std::span<const std::string_view>{views}, config);
    gTrackAllocations = false;

    std::ostringstream json;
    json << "{\"benchmark\":\"simeon_concept_mining_alloc\",\"doc_count\":" << docCount
         << ",\"vocabulary_size\":" << vocabularySize << ",\"concept_count\":" << concepts.size()
         << ",\"allocation_count\":" << gAllocationCount
         << ",\"allocation_bytes\":" << gAllocationBytes
         << ",\"output_fingerprint\":" << simeon::detail::ConceptIndexFingerprint::compute(concepts)
         << "}";
#elif SIMEON_SAMPLE_RSS
    enum class SamplingState { Idle, Collecting, Stop };
    std::mutex samplerMutex;
    std::condition_variable samplerCv;
    SamplingState samplerState = SamplingState::Idle;
    bool samplerReady = false;
    std::uint64_t peakMiningRss = 0;
    std::thread sampler([&] {
        std::unique_lock lock{samplerMutex};
        samplerReady = true;
        samplerCv.notify_all();
        while (samplerState != SamplingState::Stop) {
            samplerCv.wait(lock, [&] { return samplerState != SamplingState::Idle; });
            while (samplerState == SamplingState::Collecting) {
                lock.unlock();
                const auto rss = currentRssBytes();
                lock.lock();
                // If collection closed while currentRssBytes() ran, discard
                // that sample. Closing the gate takes this same lock and thus
                // acknowledges every in-flight sample before returning.
                if (samplerState == SamplingState::Collecting)
                    peakMiningRss = std::max(peakMiningRss, rss);
                samplerCv.wait_for(lock, std::chrono::milliseconds(1),
                                   [&] { return samplerState != SamplingState::Collecting; });
            }
        }
    });

    std::unique_lock samplerLock{samplerMutex};
    samplerCv.wait(samplerLock, [&] { return samplerReady; });
    const auto rssBefore = currentRssBytes();
    peakMiningRss = rssBefore;
    samplerState = SamplingState::Collecting;
    samplerCv.notify_all();
    samplerLock.unlock();

    auto concepts = simeon::mine_concepts(index, std::span<const std::string_view>{views}, config);

    samplerLock.lock();
    samplerState = SamplingState::Idle;
    samplerCv.notify_all();
    samplerState = SamplingState::Stop;
    samplerCv.notify_all();
    const auto peakRss = peakMiningRss;
    samplerLock.unlock();
    sampler.join();
    const auto rssAfter = currentRssBytes();
    const auto peakGrowth = peakRss > rssBefore ? peakRss - rssBefore : 0;

    std::ostringstream json;
    json << "{\"benchmark\":\"simeon_concept_mining_rss\",\"doc_count\":" << docCount
         << ",\"vocabulary_size\":" << vocabularySize << ",\"concept_count\":" << concepts.size()
         << ",\"rss_before_bytes\":" << rssBefore << ",\"rss_after_bytes\":" << rssAfter
         << ",\"peak_mining_rss_bytes\":" << peakRss
         << ",\"peak_mining_growth_bytes\":" << peakGrowth
         << ",\"output_fingerprint\":" << simeon::detail::ConceptIndexFingerprint::compute(concepts)
         << "}";
#else
    const auto start = std::chrono::steady_clock::now();
    auto concepts = simeon::mine_concepts(index, std::span<const std::string_view>{views}, config);
    const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();

    std::ostringstream json;
    json << "{\"benchmark\":\"simeon_concept_mining\",\"doc_count\":" << docCount
         << ",\"vocabulary_size\":" << vocabularySize << ",\"concept_count\":" << concepts.size()
         << ",\"elapsed_us\":" << elapsedUs
         << ",\"output_fingerprint\":" << simeon::detail::ConceptIndexFingerprint::compute(concepts)
         << "}";
#endif

    return writeRecord(json.str(), argc, argv) ? 0 : 2;
}

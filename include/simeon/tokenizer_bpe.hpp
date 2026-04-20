#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "simeon/tokenizer.hpp"

namespace simeon {

// Byte-level BPE (Sennrich, Haddow & Birch 2016) computed by iterated
// byte-pair counting. simeon ships no built-in merges table — the caller
// learns one from a domain-specific seed corpus and passes it in via
// TokenizerConfig::bpe. Keeps the binary identical regardless of target
// language, and keeps "training-free" honest: no gradient descent is
// involved, only corpus-statistics counting.
//
// Encoding a word is greedy lowest-rank: at each step apply the merge with
// the smallest merge-id (oldest merge first), repeat until no applicable
// merge remains. This matches huggingface tokenizers' BPE behaviour.
class BpeMerges {
public:
    // Counts byte-pair frequencies across whitespace-split tokens of every
    // input text and iteratively merges the most-frequent pair until either
    // `target_vocab_size` is reached (counting the 256 byte initial vocab)
    // or no pair has frequency > 1. Deterministic given a fixed seed_corpus
    // and target_vocab_size.
    static BpeMerges learn(std::span<const std::string_view> seed_corpus,
                           std::uint32_t target_vocab_size);

    // Convenience: learn from a single text by treating each whitespace-
    // separated chunk as one corpus token.
    static BpeMerges learn_from_text(std::string_view text,
                                     std::uint32_t target_vocab_size);

    // Round-trippable text format. One line per merge:
    //   `<left token bytes>\t<right token bytes>\n`
    // Token bytes are emitted with non-printable / whitespace bytes escaped
    // as \xNN. serialize() / from_text() are bit-for-bit inverses.
    std::string serialize() const;
    static BpeMerges from_text(std::string_view serialized);

    // Encode a single word's bytes into subword tokens and emit each via
    // sink.on_token(token, 1.0f). For words with no applicable merges this
    // emits one token per byte, which stays correct (every byte is a vocab
    // member at id < 256).
    void apply(std::string_view word, NGramEmitter& sink) const;

    std::uint32_t vocab_size() const noexcept {
        return static_cast<std::uint32_t>(vocab_.size());
    }
    std::uint32_t merge_count() const noexcept {
        return static_cast<std::uint32_t>(vocab_.size() > 256 ? vocab_.size() - 256 : 0);
    }

private:
    BpeMerges();

    // vocab_[i] is the byte string for token id i. Ids 0..255 are the single
    // bytes (initial alphabet); ids ≥ 256 are merged tokens, in the order
    // they were learned (so merge id == merge rank).
    std::vector<std::string> vocab_;
    // pair (left_id << 32 | right_id) -> merged token id.
    std::unordered_map<std::uint64_t, std::uint32_t> pair_to_id_;
};

}  // namespace simeon

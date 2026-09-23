#pragma once

// Locale-independent ASCII character classes. They match <cctype> under the "C" locale, so
// tokenization and hashing stay identical no matter what LC_CTYPE the host application sets;
// bytes >= 0x80 are never letters, digits, or spaces and are never case-folded.

namespace simeon::detail {

[[nodiscard]] constexpr bool ascii_isdigit(unsigned char c) noexcept {
    return c >= '0' && c <= '9';
}

[[nodiscard]] constexpr bool ascii_isupper(unsigned char c) noexcept {
    return c >= 'A' && c <= 'Z';
}

[[nodiscard]] constexpr bool ascii_islower(unsigned char c) noexcept {
    return c >= 'a' && c <= 'z';
}

[[nodiscard]] constexpr bool ascii_isalpha(unsigned char c) noexcept {
    return ascii_isupper(c) || ascii_islower(c);
}

[[nodiscard]] constexpr bool ascii_isalnum(unsigned char c) noexcept {
    return ascii_isalpha(c) || ascii_isdigit(c);
}

[[nodiscard]] constexpr bool ascii_isspace(unsigned char c) noexcept {
    return c == ' ' || (c >= '\t' && c <= '\r');
}

[[nodiscard]] constexpr unsigned char ascii_tolower(unsigned char c) noexcept {
    return ascii_isupper(c) ? static_cast<unsigned char>(c - 'A' + 'a') : c;
}

} // namespace simeon::detail

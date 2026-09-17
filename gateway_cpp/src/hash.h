// base64 + SHA-1 (RFC 6455 handshake needs both)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

std::string b64_encode(const void* data, size_t n);
std::vector<uint8_t> b64_decode(const std::string& s);

// FIPS 180-1 SHA-1; returns 20 raw bytes.
std::array<uint8_t, 20> sha1(const uint8_t* data, size_t n);
std::string sha1_b64(const std::string& key);

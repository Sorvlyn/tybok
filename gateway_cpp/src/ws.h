// Minimal RFC 6455 WebSocket *server*: HTTP upgrade handshake (incl. a tiny
// GET /health), frame read/write with masking + fragmentation, ping/pong/close.
// One thread per connection; blocking sockets.
#pragma once

#include <cstdint>
#include <string>

namespace ws {

// Read + answer the HTTP request on `fd`. Returns false if the socket is not a
// valid WebSocket upgrade (also answers GET /health with a 200 JSON body).
// On success the connection is upgraded and ready for frames.
bool handshake(int fd, const std::string& health_json, std::string* err);

// Read one complete WebSocket message (reassembles continuation frames).
// `opcode` receives 1 (text) / 2 (binary) / 8 (close). Ping frames are answered
// automatically with pong. Returns false on protocol error / EOF / close.
bool read_message(int fd, std::string* payload, uint8_t* opcode, std::string* err);

bool send_text(int fd, const std::string& payload, std::string* err);
bool send_close(int fd, uint16_t code, std::string* err);

}  // namespace ws

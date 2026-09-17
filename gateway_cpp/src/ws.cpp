#include "ws.h"

#include "hash.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace ws {

static constexpr size_t kMaxMessage = 128u << 20;  // 128 MiB

static bool read_exact(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < n) {
        ssize_t r = ::recv(fd, p + done, n - done, 0);
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

static bool write_all(int fd, const char* data, size_t n) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = ::send(fd, data + done, n - done, MSG_NOSIGNAL);
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

bool handshake(int fd, const std::string& health_json, std::string* err) {
    // read request headers
    std::string req;
    char buf[4096];
    while (req.find("\r\n\r\n") == std::string::npos) {
        ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
        if (r <= 0) {
            if (err) *err = "connection closed during handshake";
            return false;
        }
        req.append(buf, static_cast<size_t>(r));
        if (req.size() > 64u << 10) {
            if (err) *err = "request headers too large";
            return false;
        }
    }

    // request line + headers
    size_t first_line_end = req.find("\r\n");
    std::string request_line = req.substr(0, first_line_end);
    std::string path = "/";
    {
        size_t sp1 = request_line.find(' ');
        size_t sp2 = sp1 == std::string::npos ? std::string::npos : request_line.find(' ', sp1 + 1);
        if (sp1 != std::string::npos && sp2 != std::string::npos) {
            path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
        }
    }
    // strip query string
    size_t q = path.find('?');
    if (q != std::string::npos) path = path.substr(0, q);

    if (path == "/health") {
        char head[256];
        int hl = std::snprintf(head, sizeof(head),
                               "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                               "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                               health_json.size());
        if (hl < 0 || !write_all(fd, head, static_cast<size_t>(hl))) {
            if (err) *err = "health write failed";
            return false;
        }
        write_all(fd, health_json.data(), health_json.size());
        ::close(fd);
        return false;  // connection done
    }

    // Sec-WebSocket-Key
    std::string key;
    size_t pos = 0;
    while ((pos = req.find("Sec-WebSocket-Key:", pos)) != std::string::npos) {
        size_t line_end = req.find("\r\n", pos);
        std::string line = req.substr(pos + 18, line_end - pos - 18);
        // trim
        size_t s = line.find_first_not_of(" \t");
        size_t e = line.find_last_not_of(" \t");
        if (s != std::string::npos) key = line.substr(s, e - s + 1);
        break;
    }
    if (key.empty()) {
        if (err) *err = "missing Sec-WebSocket-Key";
        ::close(fd);
        return false;
    }

    std::string accept = sha1_b64(key);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\n"
                       "Upgrade: websocket\r\n"
                       "Connection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
    if (!write_all(fd, resp.data(), resp.size())) {
        if (err) *err = "upgrade write failed";
        return false;
    }
    return true;
}

static bool send_frame(int fd, uint8_t opcode, const std::string& payload, std::string* err) {
    std::string frame;
    size_t n = payload.size();
    frame.push_back(static_cast<char>(0x80 | opcode));
    if (n < 126) {
        frame.push_back(static_cast<char>(n));
    } else if (n < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<char>((n >> 8) & 0xFF));
        frame.push_back(static_cast<char>(n & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--) frame.push_back(static_cast<char>((n >> (8 * i)) & 0xFF));
    }
    frame += payload;
    return write_all(fd, frame.data(), frame.size());
}

bool send_text(int fd, const std::string& payload, std::string* err) {
    return send_frame(fd, 0x1, payload, err);
}

bool send_close(int fd, uint16_t code, std::string* err) {
    std::string payload;
    payload.push_back(static_cast<char>((code >> 8) & 0xFF));
    payload.push_back(static_cast<char>(code & 0xFF));
    return send_frame(fd, 0x8, payload, err);
}

bool read_message(int fd, std::string* payload, uint8_t* opcode, std::string* err) {
    payload->clear();
    uint8_t msg_opcode = 0;
    bool first = true;
    for (;;) {
        uint8_t h[2];
        if (!read_exact(fd, h, 2)) {
            if (err) *err = "EOF";
            return false;
        }
        bool fin = (h[0] & 0x80) != 0;
        uint8_t op = h[0] & 0x0F;
        bool masked = (h[1] & 0x80) != 0;
        uint64_t len = h[1] & 0x7F;
        if (len == 126) {
            uint8_t ext[2];
            if (!read_exact(fd, ext, 2)) return false;
            len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
        } else if (len == 127) {
            uint8_t ext[8];
            if (!read_exact(fd, ext, 8)) return false;
            len = 0;
            for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
        }
        if (len > kMaxMessage) {
            if (err) *err = "frame too large";
            return false;
        }
        uint8_t mask[4] = {0, 0, 0, 0};
        if (masked && !read_exact(fd, mask, 4)) return false;

        std::string data;
        data.resize(static_cast<size_t>(len));
        if (len && !read_exact(fd, data.data(), static_cast<size_t>(len))) return false;
        if (masked) {
            for (size_t i = 0; i < data.size(); i++) data[i] ^= mask[i & 3];
        }

        if (op == 0x9) {  // ping -> pong
            send_frame(fd, 0xA, data, nullptr);
            continue;
        }
        if (op == 0xA) continue;  // pong
        if (op == 0x8) {  // close
            send_frame(fd, 0x8, data, nullptr);
            if (err) *err = "close";
            return false;
        }
        if (op == 0x0) {  // continuation
            if (first) {
                if (err) *err = "unexpected continuation";
                return false;
            }
            *payload += data;
        } else if (op == 0x1 || op == 0x2) {
            if (!first) {
                if (err) *err = "new data frame during fragmented message";
                return false;
            }
            msg_opcode = op;
            *payload += data;
        } else {
            if (err) *err = "unknown opcode";
            return false;
        }
        first = false;
        if (fin) {
            *opcode = msg_opcode;
            return true;
        }
    }
}

}  // namespace ws

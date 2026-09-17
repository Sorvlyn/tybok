#include "worker.h"

#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
constexpr uint32_t kPrefixBytes = 8;  // total_len + header_len
}

WorkerClient::~WorkerClient() {
    if (fd_ >= 0) ::close(fd_);
}

bool WorkerClient::connect(const std::string& socket_path, std::string* err) {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        if (err) *err = "socket() failed";
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        if (err) *err = "socket path too long";
        ::close(fd);
        return false;
    }
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (err) *err = "connect(" + socket_path + ") failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }
    fd_ = fd;
    return true;
}

bool WorkerClient::recv_body(std::string* body) {
    uint8_t prefix[8];
    size_t got = 0;
    while (got < 8) {
        ssize_t r = ::recv(fd_, prefix + got, 8 - got, 0);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    uint32_t total = 0, hlen = 0;
    for (int i = 0; i < 4; i++) {
        total |= static_cast<uint32_t>(prefix[i]) << (8 * i);
        hlen |= static_cast<uint32_t>(prefix[4 + i]) << (8 * i);
    }
    if (total < kPrefixBytes + hlen || hlen > (1u << 20)) return false;
    body->resize(hlen);
    size_t done = 0;
    while (done < hlen) {
        ssize_t r = ::recv(fd_, body->data() + done, hlen - done, 0);
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

bool WorkerClient::describe(json::Value* out, std::string* err) {
    std::lock_guard<std::mutex> lock(mu_);
    json::Value req = json::obj({
        {"id", json::str("gw-describe")},
        {"type", json::str("describe")},
    });
    if (!send_frame(json::dump(req), "")) {
        if (err) *err = "worker send failed";
        return false;
    }
    std::string body;
    if (!recv_body(&body)) {
        if (err) *err = "worker recv failed";
        return false;
    }
    json::Value resp = json::parse(body, err);
    if (!err->empty() || !resp.is_object()) return false;
    const json::Value* ok = resp.find("ok");
    if (!ok || !ok->as_bool()) {
        if (err) *err = "worker describe error: " + (resp.find("error") ? resp.find("error")->as_string() ? *resp.find("error")->as_string() : "?" : "?");
        return false;
    }
    const json::Value* data = resp.find("data");
    if (!data) {
        if (err) *err = "describe missing data";
        return false;
    }
    *out = *data;
    return true;
}

bool WorkerClient::send_frame(const std::string& header, const std::string& payload) {
    uint32_t total = kPrefixBytes + static_cast<uint32_t>(header.size() + payload.size());
    uint32_t hlen = static_cast<uint32_t>(header.size());
    uint8_t prefix[8];
    for (int i = 0; i < 4; i++) {
        prefix[i] = static_cast<uint8_t>((total >> (8 * i)) & 0xFF);
        prefix[4 + i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xFF);
    }
    if (::send(fd_, prefix, 8, MSG_NOSIGNAL) != 8) return false;
    size_t off = 0;
    while (off < header.size()) {
        ssize_t r = ::send(fd_, header.data() + off, header.size() - off, MSG_NOSIGNAL);
        if (r <= 0) return false;
        off += static_cast<size_t>(r);
    }
    off = 0;
    while (off < payload.size()) {
        ssize_t r = ::send(fd_, payload.data() + off, payload.size() - off, MSG_NOSIGNAL);
        if (r <= 0) return false;
        off += static_cast<size_t>(r);
    }
    return true;
}

bool WorkerClient::request(const std::string& header_json, json::Value* resp, std::string* raw_body, std::string* err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!send_frame(header_json, "")) {
        if (err) *err = "worker send failed";
        return false;
    }
    std::string body;
    if (!recv_body(&body)) {
        if (err) *err = "worker recv failed";
        return false;
    }
    *resp = json::parse(body, err);
    if (raw_body) *raw_body = std::move(body);
    if (!err->empty() || !resp->is_object()) return false;
    return true;
}

bool WorkerClient::request_payload(const std::string& header_json, const std::string& payload,
                                   json::Value* resp, std::string* raw_body, std::string* err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!send_frame(header_json, payload)) {
        if (err) *err = "worker send failed";
        return false;
    }
    std::string body;
    if (!recv_body(&body)) {
        if (err) *err = "worker recv failed";
        return false;
    }
    *resp = json::parse(body, err);
    if (raw_body) *raw_body = std::move(body);
    if (!err->empty() || !resp->is_object()) return false;
    return true;
}

// Worker IPC client: one persistent unix-socket connection to the Python
// inference worker, speaking the same length-prefixed frame protocol as the
// Python gateway (protocol.py): [4B LE total][4B LE header_len][JSON header].
// Requests are serialized with a mutex (the worker's inference is serial too).
#pragma once

#include <mutex>
#include <string>

#include "json.h"

class WorkerClient {
  public:
    ~WorkerClient();
    bool connect(const std::string& socket_path, std::string* err);
    // send a describe control frame; on success *out = the response's "data"
    bool describe(json::Value* out, std::string* err);
    // send an infer request (header JSON) and parse the response; `raw_body`
    // (optional) receives the raw response text, which the response's raw
    // spans (Value::raw_begin/raw_end) point into -- copy spans before it dies
    bool request(const std::string& header_json, json::Value* resp, std::string* raw_body, std::string* err);
    // same, with a binary tensor payload appended after the JSON header -- the
    // byte transport's frame layout ([4B total][4B header_len][JSON][payload],
    // identical to protocol.py::encode_request), used when GPU-direct is off
    bool request_payload(const std::string& header_json, const std::string& payload,
                         json::Value* resp, std::string* raw_body, std::string* err);

  private:
    bool send_frame(const std::string& header, const std::string& payload);
    bool recv_body(std::string* body);
    int fd_ = -1;
    std::mutex mu_;
};

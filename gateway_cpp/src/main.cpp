// C++ gateway: WebSocket front-end + fused decode->resize->encode pipeline +
// GPU-direct IPC to the Python inference worker (drop-in for gateway.py).
//
// Layout: python worker --socket /tmp/tybok_worker.sock  (unchanged)
//         tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765
//
// Threading / pipelining:
//   - main thread: accept loop
//   - per connection: a READER thread (WS recv into a bounded queue) + the
//     processing loop (decode dispatch -> GPU -> worker roundtrip -> respond)
//   - a thread pool runs the per-camera decode+resize jobs in parallel
//   - lookahead: the processing loop dispatches the NEXT message's decode jobs
//     (from the queue) BEFORE the current worker roundtrip, so the next frame's
//     decode+resize overlaps the current inference -- the same cross-request
//     pipelining the Python gateway gets from --max-inflight.
#include <arpa/inet.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <netinet/in.h>
#include <optional>
#include <queue>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gpu_ipc.h"
#include "hash.h"
#include "image.h"
#include "json.h"
#include "log.h"
#include "worker.h"
#include "ws.h"

// ------------------------------------------------------------------ thread pool
class ThreadPool {
  public:
    explicit ThreadPool(size_t n) {
        for (size_t i = 0; i < n; i++) threads_.emplace_back([this] { worker_loop(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    template <class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            jobs_.emplace_back([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

  private:
    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }
    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> jobs_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// The observation prefix every camera slot lives under on the worker side. Backends report
// their slots either bare (``camera1``) or already prefixed (fastwam's
// ``observation.images.image``), so both forms are normalised through these helpers.
static constexpr char kObsImages[] = "observation.images.";

static bool has_obs_prefix(const std::string& s) { return s.rfind(kObsImages, 0) == 0; }

// Worker-side tensor name for a ``describe()`` camera entry.
static std::string slot_key(const std::string& cam) {
    return has_obs_prefix(cam) ? cam : std::string(kObsImages) + cam;
}

// Bare slot name: the shorthand a client may send instead of the reported key.
static std::string bare_slot(const std::string& cam) {
    return has_obs_prefix(cam) ? cam.substr(sizeof(kObsImages) - 1) : cam;
}

// ------------------------------------------------------------------ context
struct Spec {
    std::vector<std::string> cameras;  // as describe() reports them: [camera1, ...] or
                                       // [observation.images.image, ...] (fastwam)
    int target_w = 512;
    int target_h = 512;
    std::string model_type = "smolvla";
};

// a parsed client message, before the (pool-parallel) image work
struct PendingRequest {
    std::map<std::string, std::string> cam_b64;  // camera -> base64 image
    std::vector<float> state;
    std::string mode = "select_action";
    std::string task;
    std::string noise;
    bool valid = false;
};

// one camera's decoded+resized result
struct CamResult {
    std::string name;
    std::shared_ptr<FloatImage> img;
    std::string err;
};

// gathered results of a request's camera jobs
struct DecodedCams {
    std::vector<CamResult> ok_cams;
    std::vector<std::string> missing;
    std::string error;  // fatal (no usable images / decode failure)
};

struct Context {
    WorkerClient worker;
    GpuIpc ring;
    std::unique_ptr<ThreadPool> pool;
    Spec spec;
    std::string health_json;
    std::atomic<long long> req_counter{0};
    bool timing = false;
    bool gpu_direct = false;  // --gpu-direct: HtoD + cudaIpcMemHandle (opt-in)

    std::string error_response(const std::string& msg) const {
        return "{\"ok\":false,\"error\":" + json::dump(json::str(msg)) + "}";
    }

    static json::Value gpu_json(const GpuTensorMeta& m) {
        json::Value::Array shape;
        for (auto s : m.shape) shape.push_back(json::numi(s));
        json::Value sv;
        sv.v = shape;
        return json::obj({
            {"device", json::numi(m.device)},
            {"handle", json::str(m.handle_b64)},
            {"size_bytes", json::numi(m.size_bytes)},
            {"offset_bytes", json::numi(m.offset_bytes)},
            {"ref_h", json::str(m.ref_h_b64)},
            {"ref_o", json::numi(m.ref_o)},
            {"ev_h", json::str(m.ev_h_b64)},
            {"ev_sync", json::boolean(true)},
            {"shape", std::move(sv)},
        });
    }

    PendingRequest parse_request(const std::string& msg, std::string* jerr) const {
        PendingRequest pr;
        json::Value req = json::parse(msg, jerr);
        if (!jerr->empty() || !req.is_object()) return pr;
        const json::Value* images = req.find("images");
        const json::Value* state_v = req.find("state");
        const json::Value* task_v = req.find("task");
        const json::Value* mode_v = req.find("mode");
        const json::Value* noise_v = req.find("noise");
        if (task_v && task_v->is_string()) pr.task = *task_v->as_string();
        if (mode_v && mode_v->is_string()) pr.mode = *mode_v->as_string();
        if (noise_v && noise_v->is_string()) pr.noise = *noise_v->as_string();
        if (state_v && state_v->is_array()) {
            for (const auto& x : *state_v->as_array()) pr.state.push_back(static_cast<float>(x.as_number()));
        }
        const json::Value::Object* img_obj = images ? images->as_object() : nullptr;
        if (img_obj) {
            for (const auto& cam : spec.cameras) {
                auto it = img_obj->find(cam);
                if (it == img_obj->end() || !it->second.is_string()) {
                    // clients may send the bare slot name or the full observation key
                    const std::string bare = bare_slot(cam);
                    it = img_obj->find(bare == cam ? slot_key(cam) : bare);
                }
                if (it == img_obj->end() || !it->second.is_string()) continue;
                pr.cam_b64[cam] = *it->second.as_string();
            }
        }
        pr.valid = true;
        return pr;
    }

    // one pool job per present camera: b64 -> decode -> fused resize+pad+float
    std::vector<std::future<CamResult>> dispatch_decode(const PendingRequest& pr) const {
        std::vector<std::future<CamResult>> jobs;
        for (const auto& [cam, b64] : pr.cam_b64) {
            jobs.push_back(pool->enqueue([this, cam, b64]() -> CamResult {
                CamResult r;
                r.name = slot_key(cam);
                std::vector<uint8_t> bytes = b64_decode(b64);
                std::shared_ptr<FloatImage> img = std::make_shared<FloatImage>();
                int h = 0, w = 0;
                std::vector<uint8_t> rgb;
                std::string e;
                if (!decode_image(bytes.data(), bytes.size(), &h, &w, &rgb, &e)) {
                    r.err = e;
                    return r;
                }
                if (!resize_pad_float(rgb.data(), h, w, spec.target_w, spec.target_h, img.get())) {
                    r.err = "resize failed";
                    return r;
                }
                r.img = img;
                return r;
            }));
        }
        return jobs;
    }

    static DecodedCams gather(const PendingRequest& pr, std::vector<std::future<CamResult>>& jobs,
                              const Spec& spec) {
        DecodedCams dc;
        std::vector<CamResult> cams;
        cams.reserve(jobs.size());
        for (auto& f : jobs) cams.push_back(f.get());
        for (const auto& cam : spec.cameras) {
            const std::string key = slot_key(cam);
            bool found = false;
            for (auto& c : cams) {
                if (c.name == key && c.img) {
                    dc.ok_cams.push_back(c);
                    found = true;
                }
            }
            if (!found) dc.missing.push_back(cam);
        }
        if (dc.ok_cams.empty()) {
            std::string expect = "[";
            for (size_t i = 0; i < spec.cameras.size(); i++) {
                if (i) expect += ", ";
                expect += spec.cameras[i];
            }
            expect += "]";
            dc.error = "no images received (expected: " + expect + ")";
        }
        (void)pr;
        return dc;
    }

    // GPU-direct HtoD + IPC handles + worker roundtrip; returns the WS response
    std::string finalize(const PendingRequest& pr, const DecodedCams& dc, const std::string& id) {
        // GPU-direct source tensors must stay alive until the worker has
        // imported them, i.e. until the roundtrip below returns. Declared at
        // function scope on purpose: a request scoped inside the gpu branch
        // would be destroyed (cuMemFree) before worker.request runs and the
        // worker would open freed memory (cudaErrorInvalidValue).
        std::unique_ptr<GpuIpc::Request> req;
        json::Value tensors = json::obj({});
        auto& to = std::get<json::Value::Object>(tensors.v);
        long long offset = 0;
        std::string payload;  // byte-transport tensor payload (empty on the gpu path)
        if (gpu_direct) {
            req = ring.begin_request(nullptr);
            if (!req) return error_response("gpu begin_request failed");
            long long ref_o = 0;
            for (const auto& c : dc.ok_cams) {
                GpuTensorMeta meta;
                std::string err;
                const size_t nbytes = c.img->chw.size() * sizeof(float);
                if (!ring.put_tensor(*req, c.img->chw.data(), nbytes,
                                     {3, spec.target_h, spec.target_w}, ref_o++, &meta, &err)) {
                    return error_response("gpu put failed: " + err);
                }
                json::Value spec_v = json::obj({
                    {"shape", json::arr({json::numi(3), json::numi(spec.target_h), json::numi(spec.target_w)})},
                    {"dtype", json::str("float32")},
                    {"offset", json::numi(offset)},
                    {"nbytes", json::numi(static_cast<long long>(nbytes))},
                    {"gpu", gpu_json(meta)},
                });
                offset += static_cast<long long>(nbytes);
                to[c.name] = std::move(spec_v);
            }
            {
                // 0-size state: dummy 1-byte alloc but size_bytes=0 so the worker
                // builds an empty tensor (torch skips the IPC open for size 0)
                static const float kDummy = 0.0f;
                const size_t nbytes = pr.state.empty() ? 1 : pr.state.size() * sizeof(float);
                GpuTensorMeta meta;
                std::string err;
                if (!ring.put_tensor(*req, pr.state.empty() ? &kDummy : pr.state.data(), nbytes,
                                     {static_cast<long long>(pr.state.size())}, ref_o++, &meta, &err)) {
                    return error_response("gpu put state failed: " + err);
                }
                if (pr.state.empty()) meta.size_bytes = 0;
                json::Value spec_v = json::obj({
                    {"shape", json::arr({json::numi(static_cast<long long>(pr.state.size()))})},
                    {"dtype", json::str("float32")},
                    {"offset", json::numi(offset)},
                    {"nbytes", json::numi(static_cast<long long>(pr.state.size() * sizeof(float)))},
                    {"gpu", gpu_json(meta)},
                });
                to["observation.state"] = std::move(spec_v);
            }
        } else {
            // byte transport -- the inline-payload layout of
            // protocol.py::encode_request: per-tensor spec + the float data
            // concatenated after the JSON header. The worker's decode_request
            // builds numpy views into the frame, so the pixels the model sees
            // are bit-identical to the python gateway's byte path.
            size_t total_bytes = 0;
            for (const auto& c : dc.ok_cams) total_bytes += c.img->chw.size() * sizeof(float);
            total_bytes += pr.state.empty() ? 0 : pr.state.size() * sizeof(float);
            payload.reserve(total_bytes);
            for (const auto& c : dc.ok_cams) {
                const size_t nbytes = c.img->chw.size() * sizeof(float);
                json::Value spec_v = json::obj({
                    {"shape", json::arr({json::numi(3), json::numi(spec.target_h), json::numi(spec.target_w)})},
                    {"dtype", json::str("float32")},
                    {"offset", json::numi(offset)},
                    {"nbytes", json::numi(static_cast<long long>(nbytes))},
                });
                offset += static_cast<long long>(nbytes);
                to[c.name] = std::move(spec_v);
                payload.append(reinterpret_cast<const char*>(c.img->chw.data()), nbytes);
            }
            {
                const size_t nbytes = pr.state.empty() ? 0 : pr.state.size() * sizeof(float);
                json::Value spec_v = json::obj({
                    {"shape", json::arr({json::numi(static_cast<long long>(pr.state.size()))})},
                    {"dtype", json::str("float32")},
                    {"offset", json::numi(offset)},
                    {"nbytes", json::numi(static_cast<long long>(nbytes))},
                });
                to["observation.state"] = std::move(spec_v);
                if (nbytes) payload.append(reinterpret_cast<const char*>(pr.state.data()), nbytes);
            }
        }

        json::Value header = json::obj({
            {"id", json::str(id)},
            {"type", json::str("infer")},
            {"mode", json::str(pr.mode)},
            {"task", json::str(pr.task)},
            {"noise", pr.noise.empty() ? json::null() : json::str(pr.noise)},
            {"quantized_images", json::boolean(false)},
            {"tensors", std::move(tensors)},
        });

        auto t_worker = std::chrono::steady_clock::now();
        (void)t_worker;
        json::Value resp;
        std::string raw;
        std::string err;
        const bool sent = gpu_direct ? worker.request(json::dump(header), &resp, &raw, &err)
                                     : worker.request_payload(json::dump(header), payload, &resp, &raw, &err);
        if (!sent) {
            return error_response("worker: " + err);
        }
        const json::Value* ok = resp.find("ok");
        if (!ok || !ok->as_bool()) {
            const json::Value* e = resp.find("error");
            return error_response(e && e->as_string() ? *e->as_string() : "worker error");
        }
        const json::Value* data = resp.find("data");
        const json::Value* action = data ? data->find("action") : nullptr;
        const json::Value* shape = data ? data->find("shape") : nullptr;
        // copy the raw spans while `raw` is alive
        const std::string action_span =
            action && action->raw_begin ? std::string(action->raw_begin, action->raw_end) : "[]";
        const std::string shape_span =
            shape && shape->raw_begin ? std::string(shape->raw_begin, shape->raw_end) : "[]";
        json::Value::Array miss;
        for (auto& m : dc.missing) miss.push_back(json::str(m));
        json::Value miss_v;
        miss_v.v = miss;

        std::string out = "{\"ok\":true,\"model\":" + json::dump(json::str(spec.model_type)) +
                          ",\"mode\":" + json::dump(json::str(pr.mode)) +
                          ",\"action\":" + action_span +
                          ",\"shape\":" + shape_span +
                          ",\"missing_cameras\":" + json::dump(miss_v) + "}";
        return out;
    }
};

// ------------------------------------------------------------------ connection
// Reader thread: recv WS messages into a bounded queue (backpressure on the
// sender). The processing loop pops a message, dispatches its decode jobs to
// the pool, and -- crucially -- dispatches the NEXT message's jobs from the
// queue before the current worker roundtrip, so decode+resize overlaps
// inference (cross-request pipelining).
static void handle_connection(int fd, Context* ctx) {
    std::string err;
    if (!ws::handshake(fd, ctx->health_json, &err)) {
        ::close(fd);
        return;
    }
    std::queue<std::string> msgs;
    std::mutex mu;
    std::condition_variable cv;
    bool reader_done = false;

    std::thread reader([&] {
        for (;;) {
            std::string payload;
            uint8_t opcode = 0;
            if (!ws::read_message(fd, &payload, &opcode, &err)) break;
            if (opcode != 1) continue;
            {
                std::unique_lock<std::mutex> lock(mu);
                // bounded backlog (backpressure on the sender); the worker is
                // the bottleneck, so this rarely blocks
                cv.wait(lock, [&] { return reader_done || msgs.size() < 8; });
                msgs.push(std::move(payload));
            }
            cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            reader_done = true;
        }
        cv.notify_one();
    });

    auto pop_msg = [&](std::string* out, bool* eof) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return reader_done || !msgs.empty(); });
        if (msgs.empty()) {
            *eof = true;
            return;
        }
        *out = std::move(msgs.front());
        msgs.pop();
        *eof = false;
    };
    auto try_pop_msg = [&](std::string* out) -> bool {
        std::lock_guard<std::mutex> lock(mu);
        if (msgs.empty()) return false;
        *out = std::move(msgs.front());
        msgs.pop();
        return true;
    };

    // ---- 2-deep pipeline: decode of msg N+1 overlaps the roundtrip of N ----
    // Invariant: `cur` (+ its decode jobs) is always in hand; `nxt` (+ jobs) is
    // in hand when the client pipelines. The blocking message fetch happens
    // only AFTER the current response was sent, so the last message of a burst
    // is never held hostage by a message that does not exist yet.
    auto process = [&](PendingRequest& pr, std::vector<std::future<CamResult>>& jobs,
                       const std::chrono::steady_clock::time_point& t0, const std::string& id) -> std::string {
        auto gather_start = std::chrono::steady_clock::now();
        DecodedCams dc = Context::gather(pr, jobs, ctx->spec);
        const double decode_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gather_start).count();
        std::string resp;
        if (!dc.error.empty()) {
            resp = ctx->error_response(dc.error);
        } else {
            resp = ctx->finalize(pr, dc, id);
        }
        if (ctx->timing) {
            LOG_INFO("[timing] req=%s wait-decode=%.2fms total=%.2fms", id.c_str(), decode_ms,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        return resp;
    };

    std::string msg0;
    bool eof = false;
    pop_msg(&msg0, &eof);
    bool have_cur = !eof;
    PendingRequest cur, nxt;
    std::vector<std::future<CamResult>> cur_jobs, nxt_jobs;
    bool have_nxt = false;
    if (have_cur) {
        std::string jerr;
        cur = ctx->parse_request(msg0, &jerr);
        cur_jobs = ctx->dispatch_decode(cur);
    }
    // lookahead: grab + dispatch msg1 now so its decode runs during the first
    // process() (only when the client pipelines; non-blocking otherwise)
    {
        std::string msg1;
        if (try_pop_msg(&msg1)) {
            std::string jerr;
            nxt = ctx->parse_request(msg1, &jerr);
            nxt_jobs = ctx->dispatch_decode(nxt);
            have_nxt = true;
        }
    }

    while (have_cur) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string id = "cpp-" + std::to_string(ctx->req_counter.fetch_add(1));
        std::string resp = process(cur, cur_jobs, t0, id);
        if (!ws::send_text(fd, resp, &err)) break;

        // advance: nxt (whose decode ran during process()) becomes cur
        if (have_nxt) {
            cur = std::move(nxt);
            cur_jobs = std::move(nxt_jobs);
            have_nxt = false;
            // refill nxt without blocking (its decode overlaps the next process)
            std::string msg2;
            if (try_pop_msg(&msg2)) {
                std::string jerr;
                nxt = ctx->parse_request(msg2, &jerr);
                nxt_jobs = ctx->dispatch_decode(nxt);
                have_nxt = true;
            }
        } else {
            // no lookahead: block for the next message -- the current response
            // was already sent, so a sequential client can proceed
            std::string msg2b;
            pop_msg(&msg2b, &eof);
            if (eof) break;
            std::string jerr;
            cur = ctx->parse_request(msg2b, &jerr);
            cur_jobs = ctx->dispatch_decode(cur);
        }
    }

    reader.join();
    ::close(fd);
}

// ------------------------------------------------------------------ main
static int listen_socket(const std::string& host, int port, std::string* err) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        if (err) *err = "socket() failed";
        return -1;
    }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (host.empty() || host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            if (err) *err = "bad host";
            ::close(fd);
            return -1;
        }
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (err) *err = std::string("bind: ") + std::strerror(errno);
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 64) != 0) {
        if (err) *err = "listen failed";
        ::close(fd);
        return -1;
    }
    return fd;
}

static bool parse_spec(const json::Value& data, Spec* spec, std::string* err) {
    const json::Value* cameras = data.find("cameras");
    const json::Value* resize = data.find("resize");
    const json::Value* model = data.find("model_type");
    if (cameras && cameras->is_array()) {
        for (const auto& c : *cameras->as_array()) {
            if (c.is_string()) spec->cameras.push_back(*c.as_string());
        }
    }
    if (resize && resize->is_array() && resize->as_array()->size() >= 2) {
        const json::Value::Array& r = *resize->as_array();
        spec->target_w = static_cast<int>(r[0].as_number(512));
        spec->target_h = static_cast<int>(r[1].as_number(512));
    }
    if (model && model->is_string()) spec->model_type = *model->as_string();
    if (spec->cameras.empty()) {
        if (err) *err = "describe: no cameras in spec";
        return false;
    }
    return true;
}

// Command line this process was started with, as a copy-pasteable string.
// ``basename(argv[0])`` rather than the full path: an absolute install path adds nothing
// to what the startup log is for (mirrors the Python side's plain ``python``).
static std::string build_command(int argc, char** argv) {
    if (argc <= 0) return {};
    std::string prog = argv[0];
    const size_t slash = prog.find_last_of('/');
    if (slash != std::string::npos) prog = prog.substr(slash + 1);
    std::string out = prog;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        out += ' ';
        if (!a.empty() && a.find_first_of(" \t\n") == std::string::npos) {
            out += a;
        } else {
            out += "'" + a + "'";   // keep args with spaces unambiguous in the log
        }
    }
    return out;
}

static std::string join_cameras(const Spec& spec) {
    std::string out;
    for (size_t i = 0; i < spec.cameras.size(); i++) {
        if (i != 0) out += ',';
        out += spec.cameras[i];
    }
    return out;
}

static std::string build_health_json(const Spec& spec) {
    json::Value::Array cams;
    for (auto& c : spec.cameras) cams.push_back(json::str(c));
    json::Value cv;
    cv.v = cams;
    return json::dump(json::obj({
        {"status", json::str("ok")},
        {"model", json::str(spec.model_type)},
        {"cameras", std::move(cv)},
    }));
}

int main(int argc, char** argv) {
    std::string worker_socket = "/tmp/tybok_worker.sock";
    std::string host = "0.0.0.0";
    int port = 8765;
    int gpu_device = 0;
    bool gpu_direct = false;
    size_t threads = std::max(2u, std::thread::hardware_concurrency());
    bool timing = false;

    for (int i = 1; i < argc; i++) {
        auto need = [&](const char* name) -> std::string {
            if (i + 1 < argc) return argv[++i];
            LOG_ERROR("%s requires a value", name);
            std::exit(2);
        };
        std::string a = argv[i];
        if (a == "--worker-socket") worker_socket = need("--worker-socket");
        else if (a == "--host") host = need("--host");
        else if (a == "--port") port = std::atoi(need("--port").c_str());
        else if (a == "--gpu-device") gpu_device = std::atoi(need("--gpu-device").c_str());
        else if (a == "--gpu-direct") gpu_direct = true;
        else if (a == "--threads") threads = static_cast<size_t>(std::atoi(need("--threads").c_str()));
        else if (a == "--timing") timing = true;
        else if (a == "--help") {
            std::printf(
                "tybok_gateway_cpp: C++ gateway (fused decode+resize, GPU-direct IPC)\n"
                "  --worker-socket PATH  python worker unix socket (default /tmp/tybok_worker.sock)\n"
                "  --host HOST           bind address (default 0.0.0.0)\n"
                "  --port PORT           ws port (default 8765)\n"
                "  --gpu-direct          HtoD + cudaIpcMemHandle zero-copy IPC (opt-in; default\n"
                "                        is the byte payload transport, like the python gateway)\n"
                "  --gpu-device N        cuda device (default 0)\n"
                "  --threads N           decode/resize pool size (default hw threads)\n"
                "  --timing              print per-request phase timings to stderr\n");
            return 0;
        } else {
            LOG_ERROR("unknown arg: %s", a.c_str());
            return 2;
        }
    }
    if (threads == 0) threads = 2;

    // One command line at startup: how this gateway was started (same convention as the
    // Python side's ``[deploy] command:``).
    LOG_INFO("[deploy] command: %s", build_command(argc, argv).c_str());

    Context ctx;
    ctx.timing = timing;
    ctx.gpu_direct = gpu_direct;
    ctx.pool = std::make_unique<ThreadPool>(threads);

    // connect + describe with retry (the python worker may still be starting). Same reporting
    // convention as the Python gateway's WorkerClient: the first failure of the wait reports the
    // reason once, the remaining attempts stay silent, and the successful connect reports once.
    // The budget matches the Python side's ``protocol.WORKER_STARTUP_TIMEOUT``: an old/slow CPU
    // can spend minutes in weight loading, and the wait costs two log lines.
    constexpr double kWorkerStartupTimeoutSec = 600.0;
    std::string err;
    json::Value spec_data;
    bool connected = false;
    int failed_attempts = 0;
    const auto wait_start = std::chrono::steady_clock::now();
    for (;;) {
        err.clear();
        if (ctx.worker.connect(worker_socket, &err) && ctx.worker.describe(&spec_data, &err)) {
            connected = true;
            break;
        }
        if (failed_attempts == 0) {
            LOG_INFO("waiting for the worker to start (%s) ...", err.c_str());
        }
        ++failed_attempts;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count()
            >= kWorkerStartupTimeoutSec) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!connected) {
        LOG_ERROR("worker unreachable: %s", err.c_str());
        return 1;
    }
    if (failed_attempts > 0) {
        const double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_start).count();
        LOG_INFO("worker is up after %.1fs (%d failed attempts while starting)", waited, failed_attempts);
    }
    if (!parse_spec(spec_data, &ctx.spec, &err)) {
        LOG_ERROR("%s", err.c_str());
        return 1;
    }
    ctx.health_json = build_health_json(ctx.spec);
    if (gpu_direct) {
        if (!ctx.ring.init(gpu_device, &err)) {
            LOG_ERROR("gpu init failed: %s", err.c_str());
            return 1;
        }
    }

    int listen_fd = listen_socket(host, port, &err);
    if (listen_fd < 0) {
        LOG_ERROR("%s", err.c_str());
        return 1;
    }
    // Logged only after the socket is bound (i.e. the deployment is actually serving), the
    // same moment the Python gateway announces itself. Field names follow the Python line.
    LOG_INFO("deployment complete: model=%s cameras=%s resize=%dx%d host=%s port=%d "
             "ws=ws://%s:%d/ws health=http://%s:%d/health worker=%s gpu_direct=%s threads=%zu",
             ctx.spec.model_type.c_str(), join_cameras(ctx.spec).c_str(),
             ctx.spec.target_w, ctx.spec.target_h, host.c_str(), port,
             host.c_str(), port, host.c_str(), port, worker_socket.c_str(),
             gpu_direct ? "on" : "off", threads);

    for (;;) {
        int cfd = ::accept(listen_fd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            LOG_WARN("accept failed: %s", std::strerror(errno));
            break;
        }
        std::thread([cfd, &ctx] { handle_connection(cfd, &ctx); }).detach();
    }
    return 0;
}

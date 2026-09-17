// C++ 网关的极薄日志：与 Python 侧 `logging` 的行格式对齐
// （`[timestamp] [component] [LEVEL] message`），统一写 stderr。
//
// 只做时间戳（秒）+ 组件名 + 级别，不引入日志库、不做级别过滤/异步队列。
// 时间戳只到秒：唯一需要毫秒的是耗时，而耗时已经是各条日志里的数值。
// 每条记录先格式化到一个 buffer 再单次 fwrite，避免多线程下同一行的几次
// stdio 调用互相穿插。
//
// 用法：LOG_INFO("ready: cameras=%zu", n);  宏自带 printf 格式校验。
#ifndef TYBOK_GATEWAY_LOG_H_
#define TYBOK_GATEWAY_LOG_H_

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace gwlog {

__attribute__((format(printf, 2, 3)))
inline void emit(const char* level, const char* fmt, ...) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);

    char msg[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    char line[1152];
    const int n = std::snprintf(line, sizeof(line), "[%s] [cpp-gateway] [%s] %s\n", ts, level, msg);
    if (n > 0) {
        const size_t len = static_cast<size_t>(n) < sizeof(line) ? static_cast<size_t>(n)
                                                                 : sizeof(line) - 1;
        std::fwrite(line, 1, len, stderr);
    }
    std::fflush(stderr);   // 崩溃 / 被杀前不丢日志（与 Python 侧 StreamHandler 行为一致）
}

}  // namespace gwlog

#define LOG_INFO(...) ::gwlog::emit("INFO", __VA_ARGS__)
#define LOG_WARN(...) ::gwlog::emit("WARNING", __VA_ARGS__)
#define LOG_ERROR(...) ::gwlog::emit("ERROR", __VA_ARGS__)

#endif  // TYBOK_GATEWAY_LOG_H_

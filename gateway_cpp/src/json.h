// Minimal JSON value/parser/writer for the C++ gateway.
// We control both the worker frame header (writer) and need to parse the
// worker's response and the client's request (parser).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace json {

struct Value {
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> v;
    // raw source span (only set by parse()): used to relay number arrays
    // verbatim (e.g. the worker's action payload) without losing precision.
    const char* raw_begin = nullptr;
    const char* raw_end = nullptr;

    bool is_object() const { return std::holds_alternative<Object>(v); }
    bool is_array() const { return std::holds_alternative<Array>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }

    const Object* as_object() const { return is_object() ? &std::get<Object>(v) : nullptr; }
    const Array* as_array() const { return is_array() ? &std::get<Array>(v) : nullptr; }
    const std::string* as_string() const { return is_string() ? &std::get<std::string>(v) : nullptr; }
    double as_number(double dflt = 0.0) const { return is_number() ? std::get<double>(v) : dflt; }
    bool as_bool(bool dflt = false) const { return is_bool() ? std::get<bool>(v) : dflt; }

    const Value* find(const std::string& key) const {
        const Object* o = as_object();
        if (!o) return nullptr;
        auto it = o->find(key);
        return it == o->end() ? nullptr : &it->second;
    }
    // "a"."b" lookup
    const Value* find2(const std::string& a, const std::string& b) const {
        const Value* v1 = find(a);
        return v1 ? v1->find(b) : nullptr;
    }
};

// Parse `text`; on failure sets *err (if non-null) and returns a null value.
Value parse(const std::string& text, std::string* err = nullptr);
// Compact serialization (escapes strings; integers written without ".0").
std::string dump(const Value& v);

// helpers to build values
Value obj(std::initializer_list<std::pair<const std::string, Value>> kv);
Value arr(std::initializer_list<Value> items);
Value str(const std::string& s);
Value num(double d);
Value numi(long long n);
Value boolean(bool b);
Value null();
}  // namespace json

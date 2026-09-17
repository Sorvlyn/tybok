#include "json.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace json {

static std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string dump(const Value& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    if (v.is_string()) return escape(*v.as_string());
    if (v.is_number()) {
        double d = v.as_number();
        if (d == static_cast<long long>(d) && d >= -9e15 && d <= 9e15) {
            char buf[32];
            auto r = std::to_chars(buf, buf + sizeof(buf), static_cast<long long>(d));
            return std::string(buf, r.ptr);
        }
        char buf[40];
        auto r = std::to_chars(buf, buf + sizeof(buf), d);
        return std::string(buf, r.ptr);
    }
    if (v.is_array()) {
        const Value::Array& a = *v.as_array();
        std::string out = "[";
        for (size_t i = 0; i < a.size(); i++) {
            if (i) out += ",";
            out += dump(a[i]);
        }
        out += "]";
        return out;
    }
    const Value::Object& o = *v.as_object();
    std::string out = "{";
    size_t i = 0;
    for (const auto& [k, val] : o) {
        if (i++) out += ",";
        out += escape(k);
        out += ":";
        out += dump(val);
    }
    out += "}";
    return out;
}

// ------------------------------------------------------------------ parser
struct Parser {
    const char* p;
    const char* end;
    std::string err;

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    bool fail(const std::string& m) {
        if (err.empty()) err = m;
        return false;
    }
    Value make_null() { return Value{std::nullptr_t{}}; }

    bool parse_string(std::string* out) {
        if (p >= end || *p != '"') return fail("expected string");
        p++;
        out->clear();
        while (p < end) {
            unsigned char c = static_cast<unsigned char>(*p);
            if (c == '"') {
                p++;
                return true;
            }
            if (c == '\\') {
                p++;
                if (p >= end) return fail("bad escape");
                char e = *p++;
                switch (e) {
                    case '"': out->push_back('"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    case 'u': {
                        if (p + 4 > end) return fail("bad \\u");
                        unsigned cp = 0;
                        for (int i = 0; i < 4; i++) {
                            char h = *p++;
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= h - '0';
                            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                            else return fail("bad \\u hex");
                        }
                        // encode UTF-8 (BMP only; surrogate pairs not needed here)
                        if (cp < 0x80) out->push_back(static_cast<char>(cp));
                        else if (cp < 0x800) {
                            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: return fail("bad escape char");
                }
                continue;
            }
            if (c < 0x20) return fail("control char in string");
            out->push_back(static_cast<char>(c));
            p++;
        }
        return fail("unterminated string");
    }

    Value parse_value() {
        skip_ws();
        const char* start = p;
        if (p >= end) { fail("empty input"); return make_null(); }
        Value v;
        char c = *p;
        if (c == '{') {
            v.v = Value::Object{};
            p++;
            skip_ws();
            if (p < end && *p == '}') { p++; v.raw_begin = start; v.raw_end = p; return v; }
            while (p < end) {
                skip_ws();
                std::string key;
                if (!parse_string(&key)) return make_null();
                skip_ws();
                if (p >= end || *p != ':') { fail("expected ':'"); return make_null(); }
                p++;
                Value val = parse_value();
                if (val.is_null() && !err.empty()) return make_null();
                std::get<Value::Object>(v.v)[key] = std::move(val);
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == '}') { p++; v.raw_begin = start; v.raw_end = p; return v; }
                fail("expected ',' or '}'");
                return make_null();
            }
            fail("unterminated object");
            return make_null();
        }
        if (c == '[') {
            v.v = Value::Array{};
            p++;
            skip_ws();
            if (p < end && *p == ']') { p++; v.raw_begin = start; v.raw_end = p; return v; }
            while (p < end) {
                Value val = parse_value();
                if (val.is_null() && !err.empty()) return make_null();
                std::get<Value::Array>(v.v).push_back(std::move(val));
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == ']') { p++; v.raw_begin = start; v.raw_end = p; return v; }
                fail("expected ',' or ']'");
                return make_null();
            }
            fail("unterminated array");
            return make_null();
        }
        if (c == '"') {
            std::string s;
            if (!parse_string(&s)) return make_null();
            v.v = std::move(s);
            v.raw_begin = start;
            v.raw_end = p;
            return v;
        }
        if (c == 't') {
            if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) { p += 4; v.v = true; v.raw_begin = start; v.raw_end = p; return v; }
            fail("bad literal");
            return make_null();
        }
        if (c == 'f') {
            if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) { p += 5; v.v = false; v.raw_begin = start; v.raw_end = p; return v; }
            fail("bad literal");
            return make_null();
        }
        if (c == 'n') {
            if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) { p += 4; v.v = std::nullptr_t{}; v.raw_begin = start; v.raw_end = p; return v; }
            fail("bad literal");
            return make_null();
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            // strtod-style number (covers ints, floats, exponents, Infinity/NaN via fallback)
            const char* num_start = p;
            std::string tok;
            while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '-' || *p == '+' ||
                               *p == '.' || *p == 'e' || *p == 'E')) {
                tok.push_back(*p++);
            }
            // reject trailing garbage like "1e" (strtod would still parse 1)
            char* parsed_end = nullptr;
            double d = std::strtod(tok.c_str(), &parsed_end);
            if (parsed_end != tok.c_str() + tok.size()) {
                // keep only the valid prefix
                tok = tok.substr(0, parsed_end - tok.c_str());
                p = num_start + tok.size();
                d = std::strtod(tok.c_str(), nullptr);
            }
            v.v = d;
            v.raw_begin = start;
            v.raw_end = p;
            return v;
        }
        fail("unexpected char");
        return make_null();
    }
};

Value parse(const std::string& text, std::string* err) {
    Parser parser{text.data(), text.data() + text.size(), ""};
    Value v = parser.parse_value();
    parser.skip_ws();
    if (!parser.err.empty() || parser.p != parser.end) {
        if (err) *err = parser.err.empty() ? "trailing characters" : parser.err;
        return Value{std::nullptr_t{}};
    }
    if (err) *err = "";
    return v;
}

Value obj(std::initializer_list<std::pair<const std::string, Value>> kv) {
    Value v;
    v.v = Value::Object{};
    for (const auto& [k, val] : kv) std::get<Value::Object>(v.v)[k] = val;
    return v;
}
Value arr(std::initializer_list<Value> items) {
    Value v;
    v.v = Value::Array{items};
    return v;
}
Value str(const std::string& s) {
    Value v;
    v.v = s;
    return v;
}
Value num(double d) {
    Value v;
    v.v = d;
    return v;
}
Value numi(long long n) {
    Value v;
    v.v = static_cast<double>(n);
    return v;
}
Value boolean(bool b) {
    Value v;
    v.v = b;
    return v;
}
Value null() {
    return Value{std::nullptr_t{}};
}

}  // namespace json

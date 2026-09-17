// Standalone decode_image check with real JPEG/PNG bytes.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/image.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_decode FILE\n");
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror("open"); return 1; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) return 1;
    std::fclose(f);

    int h = 0, w = 0;
    std::vector<uint8_t> rgb;
    std::string err;
    if (!decode_image(data.data(), data.size(), &h, &w, &rgb, &err)) {
        std::fprintf(stderr, "decode failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "decoded %dx%d, %zu bytes rgb, first pixel %d,%d,%d\n",
                 w, h, rgb.size(), rgb[0], rgb[1], rgb[2]);
    return 0;
}

// Numerics check for the fused resize: reads a raw RGB uint8 input file and
// writes the (target_h, target_w) float32 CHW output, for comparison against
// torch's F.interpolate + F.pad.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/image.h"

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: test_image IN.rgb H W TARGET_W TARGET_H OUT.f32\n");
        return 2;
    }
    const char* in_path = argv[1];
    int h = std::atoi(argv[2]);
    int w = std::atoi(argv[3]);
    int tw = std::atoi(argv[4]);
    int th = std::atoi(argv[5]);
    const char* out_path = argv[6];

    FILE* f = std::fopen(in_path, "rb");
    if (!f) { std::perror("open in"); return 1; }
    std::vector<uint8_t> rgb(static_cast<size_t>(h) * w * 3);
    if (std::fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    std::fclose(f);

    FloatImage out;
    if (!resize_pad_float(rgb.data(), h, w, tw, th, &out)) {
        std::fprintf(stderr, "resize_pad_float failed\n");
        return 1;
    }
    FILE* o = std::fopen(out_path, "wb");
    if (!o) { std::perror("open out"); return 1; }
    std::fwrite(out.chw.data(), sizeof(float), out.chw.size(), o);
    std::fclose(o);
    std::fprintf(stderr, "out %dx%d, %zu floats\n", out.w, out.h, out.chw.size());
    return 0;
}

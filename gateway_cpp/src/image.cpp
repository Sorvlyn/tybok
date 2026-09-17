#include "image.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <jpeglib.h>
#include <png.h>

// ------------------------------------------------------------------ decode
struct local_err_mgr {
    jpeg_error_mgr pub;
    jmp_buf jb;
    std::string msg;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    local_err_mgr* self = reinterpret_cast<local_err_mgr*>(cinfo->err);
    char buf[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buf);
    self->msg = buf;
    longjmp(self->jb, 1);
}

static void jpeg_silent_output(j_common_ptr) {}  // warnings: no stderr spam

static bool decode_jpeg(const uint8_t* data, size_t n, int* h, int* w, std::vector<uint8_t>* rgb, std::string* err) {
    local_err_mgr lerr{};
    jpeg_decompress_struct cinfo{};
    // jpeg_std_error fills ALL five function pointers of the error manager
    // (zero-init leaves emit_message/format_message/reset_error_mgr NULL, which
    // libjpeg calls during read_header -> segfault)
    cinfo.err = jpeg_std_error(&lerr.pub);
    lerr.pub.error_exit = jpeg_error_exit;
    lerr.pub.output_message = jpeg_silent_output;
    if (setjmp(lerr.jb)) {
        if (err) *err = lerr.msg.empty() ? "jpeg decode error" : "jpeg: " + lerr.msg;
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, n);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        if (err) *err = "jpeg: bad header";
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    *w = cinfo.output_width;
    *h = cinfo.output_height;
    rgb->resize(static_cast<size_t>(*w) * *h * 3);
    size_t stride = static_cast<size_t>(*w) * 3;
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row = rgb->data() + static_cast<size_t>(cinfo.output_scanline) * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}

static bool decode_png(const uint8_t* data, size_t n, int* h, int* w, std::vector<uint8_t>* rgb, std::string* err) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, data, n)) {
        if (err) *err = std::string("png: ") + image.message;
        return false;
    }
    image.format = PNG_FORMAT_RGB;  // forces 8-bit RGB conversion
    *w = static_cast<int>(image.width);
    *h = static_cast<int>(image.height);
    rgb->resize(static_cast<size_t>(*w) * *h * 3);
    if (!png_image_finish_read(&image, nullptr, rgb->data(), 0, nullptr)) {
        if (err) *err = std::string("png: ") + image.message;
        return false;
    }
    return true;
}

bool decode_image(const uint8_t* data, size_t n, int* h, int* w, std::vector<uint8_t>* rgb, std::string* err) {
    if (n >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
        return decode_jpeg(data, n, h, w, rgb, err);
    }
    if (n >= 8 && std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0) {
        return decode_png(data, n, h, w, rgb, err);
    }
    if (err) *err = "unsupported image format (expected JPEG or PNG)";
    return false;
}

// ------------------------------------------------- fused resize + pad + /255
// Replicates torch CPU upsample_bilinear2d (align_corners=false) with float
// arithmetic in the exact same order, then F.pad((pad_w,0,pad_h,0), 0).
bool resize_pad_float(const uint8_t* rgb, int h, int w, int target_w, int target_h, FloatImage* out) {
    const int H = h, W = w;
    if (H <= 0 || W <= 0) return false;

    // ratio = max(W/tw, H/th); resized = int(src / ratio) -- mirror Python
    const double ratio = std::max(static_cast<double>(W) / target_w, static_cast<double>(H) / target_h);
    const int rh = static_cast<int>(H / ratio);
    const int rw = static_cast<int>(W / ratio);
    const int pad_h = std::max(0, target_h - rh);
    const int pad_w = std::max(0, target_w - rw);
    // guard: a pathological target could give rh/rw == 0
    if (rh <= 0 || rw <= 0) return false;

    // precompute per-output-row/col source indices + weights (torch semantics)
    // NB: torch's area_pixel_compute_source_index evaluates
    //   scale * (dst + 0.5) - 0.5   in DOUBLE (the 0.5 literals are doubles)
    // and narrows to float on return -- replicate exactly for bit-exactness.
    std::vector<int> h0p(rh), h1p(rh);
    std::vector<float> h0w(rh), h1w(rh);
    const float scale_h = static_cast<float>(H) / static_cast<float>(rh);
    for (int i = 0; i < rh; i++) {
        float src = static_cast<float>(static_cast<double>(scale_h) * (static_cast<double>(i) + 0.5) - 0.5);
        if (src < 0.0f) src = 0.0f;
        int i0 = static_cast<int>(src);
        if (i0 > H - 1) i0 = H - 1;
        int i1 = std::min(i0 + 1, H - 1);
        float lam = src - static_cast<float>(i0);
        h0p[i] = i0;
        h1p[i] = i1;
        h0w[i] = 1.0f - lam;
        h1w[i] = lam;
    }
    std::vector<int> w0p(rw), w1p(rw);
    std::vector<float> w0w(rw), w1w(rw);
    const float scale_w = static_cast<float>(W) / static_cast<float>(rw);
    for (int j = 0; j < rw; j++) {
        float src = static_cast<float>(static_cast<double>(scale_w) * (static_cast<double>(j) + 0.5) - 0.5);
        if (src < 0.0f) src = 0.0f;
        int j0 = static_cast<int>(src);
        if (j0 > W - 1) j0 = W - 1;
        int j1 = std::min(j0 + 1, W - 1);
        float lam = src - static_cast<float>(j0);
        w0p[j] = j0;
        w1p[j] = j1;
        w0w[j] = 1.0f - lam;
        w1w[j] = lam;
    }

    out->c = 3;
    out->h = target_h;
    out->w = target_w;
    out->chw.assign(static_cast<size_t>(3) * target_h * target_w, 0.0f);  // zero pad

    const size_t src_stride = static_cast<size_t>(W) * 3;
    for (int oy = 0; oy < rh; oy++) {
        const int i0 = h0p[oy];
        const int i1 = h1p[oy];
        const float h0wv = h0w[oy];
        const float h1wv = h1w[oy];
        const uint8_t* row0 = rgb + static_cast<size_t>(i0) * src_stride;
        const uint8_t* row1 = rgb + static_cast<size_t>(i1) * src_stride;
        const size_t out_row = static_cast<size_t>(oy + pad_h) * target_w + pad_w;
        for (int ox = 0; ox < rw; ox++) {
            const int j0 = w0p[ox];
            const int j1 = w1p[ox];
            const float w0wv = w0w[ox];
            const float w1wv = w1w[ox];
            for (int c = 0; c < 3; c++) {
                // torch order: /255 first (float), then the bilinear expression
                const float v00 = static_cast<float>(row0[j0 * 3 + c]) / 255.0f;
                const float v01 = static_cast<float>(row0[j1 * 3 + c]) / 255.0f;
                const float v10 = static_cast<float>(row1[j0 * 3 + c]) / 255.0f;
                const float v11 = static_cast<float>(row1[j1 * 3 + c]) / 255.0f;
                const float val = h0wv * (w0wv * v00 + w1wv * v01) + h1wv * (w0wv * v10 + w1wv * v11);
                out->chw[static_cast<size_t>(c) * target_h * target_w + out_row + static_cast<size_t>(ox)] = val;
            }
        }
    }
    return true;
}

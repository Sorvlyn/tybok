// Fused decode -> resize -> encode pipeline.
//
// decode_image: JPEG (libjpeg) / PNG (libpng) bytes -> (H,W,3) uint8 RGB,
// mirroring the Python gateway's torchvision/Pillow decode.
//
// resize_pad_float: replicates the Python gateway's prepare_image_tensor
// (image_utils.resize_with_pad) *bit-exactly*:
//   1. divide by 255 into float [0,1]  (decode step in the Python path)
//   2. aspect-preserving resize with torch F.interpolate(bilinear,
//      align_corners=False) -- same float arithmetic and op order as torch's
//      CPU upsample_bilinear2d kernel
//   3. zero-pad on LEFT and TOP to the target (width, height)
// All in a single pass over the output (fused), CHW float32 out.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FloatImage {
    int c = 3;
    int h = 0;
    int w = 0;
    std::vector<float> chw;  // c*h*w
};

// Decode JPEG or PNG bytes to RGB uint8 (H,W,3). Returns false + *err on failure.
bool decode_image(const uint8_t* data, size_t n, int* h, int* w, std::vector<uint8_t>* rgb, std::string* err);

// Fused resize+pad+normalize. target_w/target_h follow the checkpoint
// convention (width, height). `rgb` is (h, w, 3) uint8.
bool resize_pad_float(const uint8_t* rgb, int h, int w, int target_w, int target_h, FloatImage* out);

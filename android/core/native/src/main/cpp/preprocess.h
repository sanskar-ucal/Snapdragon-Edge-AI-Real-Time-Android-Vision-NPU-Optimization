#pragma once

#include <cstdint>
#include <vector>

namespace ev {

enum class OutputType { kFloat32, kFloat16 };

struct LetterboxInfo {
  float scale;
  int pad_x;
  int pad_y;
  int content_w;
  int content_h;
};

// Same math as ml/common.py::letterbox_params so host and device agree.
LetterboxInfo ComputeLetterbox(int src_w, int src_h, int dst);

// Android YUV_420_888: chroma planes may be planar (pixel_stride 1) or interleaved (2).
struct YuvFrame {
  const uint8_t* y;
  const uint8_t* u;
  const uint8_t* v;
  int width;
  int height;
  int y_row_stride;
  int uv_row_stride;
  int uv_pixel_stride;
};

uint16_t FloatToHalf(float f);

// Fused color conversion + rotation + letterbox resize + NCHW normalization in one pass.
// Index LUTs and the padding fill are cached and only rebuilt when geometry or the output
// buffer changes, so steady-state frames do no allocation and touch padding zero times.
class Preprocessor {
 public:
  // rotation_degrees is clockwise rotation needed to make the frame upright (0/90/180/270).
  LetterboxInfo FromYuv(const YuvFrame& frame, int rotation_degrees, int dst, OutputType type, void* out);
  LetterboxInfo FromRgba(const uint8_t* rgba, int width, int height, int row_stride, int dst, OutputType type,
                         void* out);

 private:
  struct Key {
    int src_w, src_h, rotation, dst;
    OutputType type;
    void* out;
    bool operator==(const Key& o) const {
      return src_w == o.src_w && src_h == o.src_h && rotation == o.rotation && dst == o.dst && type == o.type &&
             out == o.out;
    }
  };

  void Prepare(const Key& key);

  Key key_{0, 0, -1, 0, OutputType::kFloat32, nullptr};
  LetterboxInfo lb_{};
  std::vector<int> col_src_;  // per output column in content area: source coordinate
  std::vector<int> row_src_;  // per output row in content area: source coordinate
};

}  // namespace ev

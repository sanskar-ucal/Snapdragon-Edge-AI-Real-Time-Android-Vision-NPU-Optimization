#include "preprocess.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace ev {
namespace {

constexpr uint8_t kPadValue = 114;

uint8_t Clamp(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

struct NormTables {
  std::array<float, 256> f32;
  std::array<uint16_t, 256> f16;
  NormTables() {
    for (int i = 0; i < 256; ++i) {
      f32[i] = static_cast<float>(i) / 255.0f;
      f16[i] = FloatToHalf(f32[i]);
    }
  }
};

const NormTables& Tables() {
  static const NormTables tables;
  return tables;
}

struct YuvSampler {
  const YuvFrame& f;
  // Full-range BT.601 (JFIF), the CameraX default, in 8.8 fixed point.
  inline void Sample(int sx, int sy, uint8_t* rgb) const {
    const int y = f.y[sy * f.y_row_stride + sx];
    const int uv = (sy >> 1) * f.uv_row_stride + (sx >> 1) * f.uv_pixel_stride;
    const int u = f.u[uv] - 128;
    const int v = f.v[uv] - 128;
    rgb[0] = Clamp(y + ((359 * v) >> 8));
    rgb[1] = Clamp(y - ((88 * u + 183 * v) >> 8));
    rgb[2] = Clamp(y + ((454 * u) >> 8));
  }
};

struct RgbaSampler {
  const uint8_t* data;
  int stride;
  inline void Sample(int sx, int sy, uint8_t* rgb) const {
    const uint8_t* p = data + sy * stride + sx * 4;
    rgb[0] = p[0];
    rgb[1] = p[1];
    rgb[2] = p[2];
  }
};

template <typename T>
inline T Normalize(uint8_t v);
template <>
inline float Normalize<float>(uint8_t v) { return Tables().f32[v]; }
template <>
inline uint16_t Normalize<uint16_t>(uint8_t v) { return Tables().f16[v]; }

// Nearest-neighbour sampling through precomputed index LUTs. For 90/270 rotations the output
// columns walk the source rows, so the roles of the two LUTs swap.
template <bool kTransposed, typename Sampler, typename T>
void Kernel(const Sampler& s, const LetterboxInfo& lb, int dst, const int* col, const int* row, T* out) {
  const size_t plane = static_cast<size_t>(dst) * dst;
  T* r = out;
  T* g = out + plane;
  T* b = out + 2 * plane;
  uint8_t rgb[3];
  for (int cy = 0; cy < lb.content_h; ++cy) {
    size_t o = static_cast<size_t>(cy + lb.pad_y) * dst + lb.pad_x;
    const int rs = row[cy];
    for (int cx = 0; cx < lb.content_w; ++cx, ++o) {
      if (kTransposed) {
        s.Sample(rs, col[cx], rgb);
      } else {
        s.Sample(col[cx], rs, rgb);
      }
      r[o] = Normalize<T>(rgb[0]);
      g[o] = Normalize<T>(rgb[1]);
      b[o] = Normalize<T>(rgb[2]);
    }
  }
}

template <typename Sampler>
void Dispatch(const Sampler& s, bool transposed, OutputType type, const LetterboxInfo& lb, int dst, const int* col,
              const int* row, void* out) {
  if (type == OutputType::kFloat32) {
    auto* o = static_cast<float*>(out);
    transposed ? Kernel<true>(s, lb, dst, col, row, o) : Kernel<false>(s, lb, dst, col, row, o);
  } else {
    auto* o = static_cast<uint16_t*>(out);
    transposed ? Kernel<true>(s, lb, dst, col, row, o) : Kernel<false>(s, lb, dst, col, row, o);
  }
}

}  // namespace

uint16_t FloatToHalf(float f) {
#if defined(__aarch64__)
  __fp16 h = static_cast<__fp16>(f);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
#else
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);  // inputs here are in [0, 1]; flush subnormals
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  mant += 0x1000u;  // round to nearest
  if (mant & 0x800000u) {
    mant = 0;
    ++exp;
  }
  return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
#endif
}

LetterboxInfo ComputeLetterbox(int src_w, int src_h, int dst) {
  const float scale = std::min(static_cast<float>(dst) / src_w, static_cast<float>(dst) / src_h);
  const int new_w = static_cast<int>(src_w * scale + 0.5f);
  const int new_h = static_cast<int>(src_h * scale + 0.5f);
  return {scale, (dst - new_w) / 2, (dst - new_h) / 2, new_w, new_h};
}

void Preprocessor::Prepare(const Key& key) {
  if (key == key_) return;
  key_ = key;
  const bool transposed = key.rotation == 90 || key.rotation == 270;
  const int rw = transposed ? key.src_h : key.src_w;
  const int rh = transposed ? key.src_w : key.src_h;
  lb_ = ComputeLetterbox(rw, rh, key.dst);

  col_src_.resize(lb_.content_w);
  row_src_.resize(lb_.content_h);
  for (int cx = 0; cx < lb_.content_w; ++cx) {
    const int rx = std::min(static_cast<int>((cx + 0.5f) / lb_.scale), rw - 1);
    switch (key.rotation) {
      case 90: col_src_[cx] = key.src_h - 1 - rx; break;   // -> source y
      case 180: col_src_[cx] = key.src_w - 1 - rx; break;  // -> source x
      default: col_src_[cx] = rx; break;                   // 0: source x, 270: source y
    }
  }
  for (int cy = 0; cy < lb_.content_h; ++cy) {
    const int ry = std::min(static_cast<int>((cy + 0.5f) / lb_.scale), rh - 1);
    switch (key.rotation) {
      case 180: row_src_[cy] = key.src_h - 1 - ry; break;  // -> source y
      case 270: row_src_[cy] = key.src_w - 1 - ry; break;  // -> source x
      default: row_src_[cy] = ry; break;                   // 0: source y, 90: source x
    }
  }

  // Padding never changes for a given geometry/buffer, so it is written once here.
  const size_t count = static_cast<size_t>(key.dst) * key.dst * 3;
  if (key.type == OutputType::kFloat32) {
    std::fill_n(static_cast<float*>(key.out), count, Tables().f32[kPadValue]);
  } else {
    std::fill_n(static_cast<uint16_t*>(key.out), count, Tables().f16[kPadValue]);
  }
}

LetterboxInfo Preprocessor::FromYuv(const YuvFrame& frame, int rotation_degrees, int dst, OutputType type,
                                    void* out) {
  Prepare({frame.width, frame.height, rotation_degrees, dst, type, out});
  const bool transposed = rotation_degrees == 90 || rotation_degrees == 270;
  Dispatch(YuvSampler{frame}, transposed, type, lb_, dst, col_src_.data(), row_src_.data(), out);
  return lb_;
}

LetterboxInfo Preprocessor::FromRgba(const uint8_t* rgba, int width, int height, int row_stride, int dst,
                                     OutputType type, void* out) {
  Prepare({width, height, 0, dst, type, out});
  Dispatch(RgbaSampler{rgba, row_stride}, false, type, lb_, dst, col_src_.data(), row_src_.data(), out);
  return lb_;
}

}  // namespace ev

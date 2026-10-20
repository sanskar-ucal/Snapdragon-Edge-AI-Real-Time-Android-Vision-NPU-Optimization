#include <cmath>
#include <cstdio>
#include <vector>

#include "preprocess.h"

namespace {

int failures = 0;

#define EXPECT(cond)                                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

bool Near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

float At(const std::vector<float>& t, int dst, int c, int x, int y) { return t[(c * dst + y) * dst + x]; }

void TestLetterboxMath() {
  const auto lb = ev::ComputeLetterbox(640, 480, 320);
  EXPECT(Near(lb.scale, 0.5f));
  EXPECT(lb.pad_x == 0 && lb.pad_y == 40);
  EXPECT(lb.content_w == 320 && lb.content_h == 240);
}

void TestRgbaColorAndPadding() {
  const int w = 64, h = 48, dst = 32;
  std::vector<uint8_t> rgba(w * h * 4);
  for (int i = 0; i < w * h; ++i) {
    rgba[i * 4] = 255;
    rgba[i * 4 + 3] = 255;
  }
  std::vector<float> out(dst * dst * 3);
  ev::Preprocessor p;
  const auto lb = p.FromRgba(rgba.data(), w, h, w * 4, dst, ev::OutputType::kFloat32, out.data());
  EXPECT(lb.pad_y == 4);
  EXPECT(Near(At(out, dst, 0, 16, 16), 1.0f) && Near(At(out, dst, 1, 16, 16), 0.0f));
  for (int c = 0; c < 3; ++c) EXPECT(Near(At(out, dst, c, 0, 0), 114.0f / 255.0f));
}

void TestInterleavedYuvGray() {
  const int w = 8, h = 8, dst = 8;
  std::vector<uint8_t> y(w * h, 128), uv(w * h / 2, 128);  // NV12-style interleaved chroma
  const ev::YuvFrame f{y.data(), uv.data(), uv.data() + 1, w, h, w, w, 2};
  std::vector<float> out(dst * dst * 3);
  ev::Preprocessor p;
  p.FromYuv(f, 0, dst, ev::OutputType::kFloat32, out.data());
  for (int c = 0; c < 3; ++c) EXPECT(Near(At(out, dst, c, 3, 3), 128.0f / 255.0f));
}

void TestRotation90() {
  // 4x2 frame, only bottom-left pixel bright. Rotating 90 deg clockwise moves it to top-left.
  const int w = 4, h = 2, dst = 4;
  std::vector<uint8_t> y(w * h, 0), u(w * h / 4, 128), v(w * h / 4, 128);
  y[1 * w + 0] = 255;
  const ev::YuvFrame f{y.data(), u.data(), v.data(), w, h, w, w / 2, 1};
  std::vector<float> out(dst * dst * 3);
  ev::Preprocessor p;
  const auto lb = p.FromYuv(f, 90, dst, ev::OutputType::kFloat32, out.data());
  EXPECT(lb.content_w == 2 && lb.content_h == 4 && lb.pad_x == 1);
  EXPECT(Near(At(out, dst, 0, lb.pad_x, 0), 1.0f));
  EXPECT(Near(At(out, dst, 0, lb.pad_x + 1, 0), 0.0f));
  EXPECT(Near(At(out, dst, 0, lb.pad_x, 3), 0.0f));
}

void TestFloat16Output() {
  EXPECT(ev::FloatToHalf(1.0f) == 0x3C00);
  EXPECT(ev::FloatToHalf(0.5f) == 0x3800);
  EXPECT(ev::FloatToHalf(0.0f) == 0x0000);
  const int w = 4, h = 4, dst = 4;
  std::vector<uint8_t> rgba(w * h * 4, 255);
  std::vector<uint16_t> out(dst * dst * 3);
  ev::Preprocessor p;
  p.FromRgba(rgba.data(), w, h, w * 4, dst, ev::OutputType::kFloat16, out.data());
  EXPECT(out[0] == 0x3C00);
}

void TestPaddingWrittenOnlyOnce() {
  const int w = 4, h = 2, dst = 4;
  std::vector<uint8_t> rgba(w * h * 4, 255);
  std::vector<float> out(dst * dst * 3);
  ev::Preprocessor p;
  p.FromRgba(rgba.data(), w, h, w * 4, dst, ev::OutputType::kFloat32, out.data());
  out[0] = -1.0f;  // a padding pixel
  p.FromRgba(rgba.data(), w, h, w * 4, dst, ev::OutputType::kFloat32, out.data());
  EXPECT(Near(out[0], -1.0f));
}

}  // namespace

int main() {
  TestLetterboxMath();
  TestRgbaColorAndPadding();
  TestInterleavedYuvGray();
  TestRotation90();
  TestFloat16Output();
  TestPaddingWrittenOnlyOnce();
  std::printf(failures ? "%d failure(s)\n" : "all preprocess tests passed\n", failures);
  return failures ? 1 : 0;
}

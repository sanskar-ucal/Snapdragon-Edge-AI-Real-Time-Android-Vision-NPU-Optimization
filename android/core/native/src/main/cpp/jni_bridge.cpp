#include <android/bitmap.h>
#include <jni.h>

#include "preprocess.h"

namespace {

ev::Preprocessor* FromHandle(jlong handle) { return reinterpret_cast<ev::Preprocessor*>(handle); }

void WriteLetterbox(JNIEnv* env, jfloatArray out, const ev::LetterboxInfo& lb) {
  const jfloat values[3] = {lb.scale, static_cast<jfloat>(lb.pad_x), static_cast<jfloat>(lb.pad_y)};
  env->SetFloatArrayRegion(out, 0, 3, values);
}

ev::OutputType Type(jboolean fp16) { return fp16 ? ev::OutputType::kFloat16 : ev::OutputType::kFloat32; }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_edgevision_core_nativeproc_NativePreprocessor_nativeCreate(JNIEnv*, jclass) {
  return reinterpret_cast<jlong>(new ev::Preprocessor());
}

JNIEXPORT void JNICALL Java_com_edgevision_core_nativeproc_NativePreprocessor_nativeDestroy(JNIEnv*, jclass,
                                                                                            jlong handle) {
  delete FromHandle(handle);
}

JNIEXPORT jboolean JNICALL Java_com_edgevision_core_nativeproc_NativePreprocessor_nativeFromYuv(
    JNIEnv* env, jclass, jlong handle, jobject y, jobject u, jobject v, jint width, jint height, jint y_row_stride,
    jint uv_row_stride, jint uv_pixel_stride, jint rotation, jint dst, jboolean fp16, jobject out,
    jfloatArray letterbox) {
  auto* yp = static_cast<const uint8_t*>(env->GetDirectBufferAddress(y));
  auto* up = static_cast<const uint8_t*>(env->GetDirectBufferAddress(u));
  auto* vp = static_cast<const uint8_t*>(env->GetDirectBufferAddress(v));
  void* op = env->GetDirectBufferAddress(out);
  const jlong needed = static_cast<jlong>(dst) * dst * 3 * (fp16 ? 2 : 4);
  if (!yp || !up || !vp || !op || env->GetDirectBufferCapacity(out) < needed) return JNI_FALSE;
  const ev::YuvFrame frame{yp, up, vp, width, height, y_row_stride, uv_row_stride, uv_pixel_stride};
  WriteLetterbox(env, letterbox, FromHandle(handle)->FromYuv(frame, rotation, dst, Type(fp16), op));
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_edgevision_core_nativeproc_NativePreprocessor_nativeFromBitmap(
    JNIEnv* env, jclass, jlong handle, jobject bitmap, jint dst, jboolean fp16, jobject out, jfloatArray letterbox) {
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
      info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    return JNI_FALSE;
  }
  void* op = env->GetDirectBufferAddress(out);
  const jlong needed = static_cast<jlong>(dst) * dst * 3 * (fp16 ? 2 : 4);
  if (!op || env->GetDirectBufferCapacity(out) < needed) return JNI_FALSE;
  void* pixels = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return JNI_FALSE;
  const auto lb = FromHandle(handle)->FromRgba(static_cast<const uint8_t*>(pixels), static_cast<int>(info.width),
                                               static_cast<int>(info.height), static_cast<int>(info.stride), dst,
                                               Type(fp16), op);
  AndroidBitmap_unlockPixels(env, bitmap);
  WriteLetterbox(env, letterbox, lb);
  return JNI_TRUE;
}

}  // extern "C"

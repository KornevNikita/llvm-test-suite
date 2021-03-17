// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// XFAIL: cuda
// XFAIL: level_zero && windows

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)

// On Windows, LevelZero returns wrong result for outside right pixel when using
// unorm_int8 data.

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured UNNORMALIZED coordinate_normalization_mode
    CLAMP address_mode and LINEAR filter_mode
*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::float4;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::float4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

// some constants.

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto unnormalized = coordinate_normalization_mode::unnormalized;
constexpr auto linear = filtering_mode::linear;

void test_unnormalized_clamp_linear_sampler(image_channel_order ChanOrder,
                                            image_channel_type ChanType) {
  int numTests = 8; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

  // we'll use these four pixels for our image. Makes it easy to measure
  // interpolation and spot "off-by-one" probs.
  // These values will work consistently with different levels of float
  // precision (like unorm_int8 vs. fp32)
  pixelT leftEdge{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT body{0.6f, 0.4f, 0.2f, 0.0f};
  pixelT bony{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT rightEdge{0.6f, 0.4f, 0.2f, 0.0f};

  queue Q;
  const sycl::range<1> ImgRange_1D(width);
  { // closure
    // - create an image
    image<1> image_1D(ChanOrder, ChanType, ImgRange_1D);
    event E_Setup = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::write>(cgh);
      cgh.single_task<class setupUnormLinear>([=]() {
        image_acc.write(0, leftEdge);
        image_acc.write(1, body);
        image_acc.write(2, bony);
        image_acc.write(3, rightEdge);
      });
    });
    E_Setup.wait();

    // use a buffer to report back test results.
    buffer<pixelT, 1> testResults((range<1>(numTests)));

    // sampler
    auto UnNorm_Clamp_Linear_sampler =
        sampler(unnormalized, addressing_mode::clamp, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_Unorm_Linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // UnNormalized Pixel Locations when using Linear Interpolation
        // (0 -------- ](1 ---------- ](2 ----------- ](3---------- ](4)

        // 0-5 read six pixels, float coordinates,   sample:   NonNormalized +
        // Clamp + Linear
        test_acc[i++] =
            image_acc.read(-1.0f, UnNorm_Clamp_Linear_sampler); // {0,0,0,0}
        test_acc[i++] = image_acc.read(
            0.0f, UnNorm_Clamp_Linear_sampler); // {0.1,0.2,0.3,0.4}
        test_acc[i++] = image_acc.read(
            1.0f, UnNorm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            2.0f, UnNorm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            3.0f, UnNorm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            4.0f,
            UnNorm_Clamp_Linear_sampler); // {0.3,0.2,0.1,0}  // interpolated?

        // 6-7 read two pixels on either side of first pixel. float coordinates.
        test_acc[i++] = image_acc.read(
            0.9999999f, UnNorm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.0000001f, UnNorm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    auto test_acc = testResults.get_access<access::mode::read>();
    for (int i = 0, idx = 0; i < numTests; i++, idx++) {
      if (i == 0) {
        idx = -1;
        std::cout << "read six pixels, float coordinates,   sample:   "
                     "NonNormalized + Clamp + Linear"
                  << std::endl;
      }
      if (i == 6) {
        idx = 1;
        std::cout << "read two pixels on either side of 1. float coordinates. "
                     "NonNormalized + Clamp + Linear"
                  << std::endl;
      }
      if (i == 7) {
        idx = 1;
      }

      pixelT testPixel = test_acc[i];
      std::cout << i << " -- " << idx << ": ";
      outputPixel(testPixel);
      std::cout << std::endl;
    }
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  if (D.has(aspect::image)) {
    // the _int8 channels are one byte per channel, or four bytes per pixel (for
    // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
    // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
    // channel, or sixteen bytes per pixel (for RGBA).

    std::cout << "fp32 -------------" << std::endl;
    test_unnormalized_clamp_linear_sampler(image_channel_order::rgba,
                                           image_channel_type::fp32);

    std::cout << "unorm_int8 -------" << std::endl;
    test_unnormalized_clamp_linear_sampler(image_channel_order::rgba,
                                           image_channel_type::unorm_int8);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: fp32 -------------
// CHECK-NEXT: read six pixels, float coordinates,   sample:   NonNormalized + Clamp + Linear
// CHECK-NEXT: 0 -- -1: {0,0,0,0} 
// CHECK-NEXT: 1 -- 0: {0.1,0.2,0.3,0.4} 
// CHECK-NEXT: 2 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 3 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 4 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 5 -- 4: {0.3,0.2,0.1,0} 
// CHECK-NEXT: read two pixels on either side of 1. float coordinates. NonNormalized + Clamp + Linear
// CHECK-NEXT: 6 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 7 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: unorm_int8 -------
// CHECK-NEXT: read six pixels, float coordinates,   sample:   NonNormalized + Clamp + Linear
// CHECK-NEXT: 0 -- -1: {0,0,0,0} 
// CHECK-NEXT: 1 -- 0: {0.1,0.2,0.3,0.4} 
// CHECK-NEXT: 2 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 3 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 4 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 5 -- 4: {0.3,0.2,0.1,0} 
// CHECK-NEXT: read two pixels on either side of 1. float coordinates. NonNormalized + Clamp + Linear
// CHECK-NEXT: 6 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 7 -- 1: {0.4,0.4,0.4,0.4}
// clang-format on
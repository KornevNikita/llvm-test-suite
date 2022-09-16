// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// Should fail because reducer copy constructor marked as delete
// (passes with 'auto &reducer')
// XFAIL: *

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  int sumResult = 0;
  {
    queue q;
    buffer<int> sumBuf { &sumResult, 1 };
    q.submit([&](sycl::handler &cgh) {
      auto sumRed = reduction(sumBuf, cgh, plus<>());
      cgh.parallel_for(range<1>{8}, sumReduction, [=](id<1> idx, auto reducer) {
        reducer++;
      });
    });
    q.wait();
  }
  return 0;
}
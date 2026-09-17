cmake_minimum_required(VERSION 3.16)
project(tybok_gateway_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(tybok_gateway_cpp
  src/main.cpp
  src/ws.cpp
  src/hash.cpp
  src/image.cpp
  src/gpu_ipc.cpp
  src/worker.cpp
  src/json.cpp
)

target_link_libraries(tybok_gateway_cpp PRIVATE
  JPEG::JPEG
  PNG::PNG
  Threads::Threads
  CUDA::cuda_driver
)

target_compile_options(tybok_gateway_cpp PRIVATE -O2 -Wall -Wextra)

# image.cpp must match torch's CPU bilinear kernel bit-for-bit, which is
# compiled with AVX2+FMA (its mul+add fuses into fma). Without these flags the
# resize differs by 1 ulp.
set_source_files_properties(src/image.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=fast")

# numerics check: fused resize vs torch (see test/test_image.cpp)
add_executable(test_image
  test/test_image.cpp
  src/image.cpp
)
target_link_libraries(test_image PRIVATE JPEG::JPEG PNG::PNG)
target_compile_options(test_image PRIVATE -O2 -Wall -Wextra)
set_source_files_properties(test/test_image.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=fast")

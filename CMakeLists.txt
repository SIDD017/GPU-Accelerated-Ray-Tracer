cmake_minimum_required(VERSION 3.8)
project(cmake_and_cuda LANGUAGES CXX CUDA)

file(GLOB_RECURSE source CONFIGURE_DEPENDS "src/*.cuh" "src/*.cu")

add_executable(CUDA_Tracer ${source})

# add_library(tracer_lib STATIC
#   tracer.cu
#   tracer.cuh
#   defines.cuh
#   )

#   add_executable(tracer_test main.cu)
#   target_link_libraries(tracer_test PRIVATE tracer_lib)
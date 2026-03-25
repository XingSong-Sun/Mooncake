file(GLOB CUDA_HETERO_SOURCES "*.cpp")

add_library(cuda_heterogeneous_transport OBJECT ${CUDA_HETERO_SOURCES})
target_link_libraries(cuda_heterogeneous_transport PRIVATE cuda cudart glog::glog)

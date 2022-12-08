cmake_minimum_required(VERSION 3.10)
project(CNNCPlus)

find_package(OpenMP REQUIRED)
message("CMAKE_CURRENT_SOURCE_DIR is " ${CMAKE_CURRENT_SOURCE_DIR})

#add_definitions(-DMOJO_AVX512)
#add_compile_options(-std=c++11 -fopenmp -O3 -mavx512f)

#execute_process(COMMAND "rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/models/snapshots/*.txt")

add_executable(TrainMnistAVX512 ${CMAKE_CURRENT_SOURCE_DIR}/examples/train_mnist.cpp)
target_include_directories(TrainMnistAVX512 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mojo)
target_link_libraries(TrainMnistAVX512 PUBLIC ${OpenMP_CXX_LIBRARIES})
target_compile_definitions(TrainMnistAVX512 PUBLIC -DMOJO_AVX512 -DMOJO_OMP)
target_compile_options(TrainMnistAVX512 PUBLIC -std=c++11 -fopenmp -O3 -mavx512f)

add_executable(TrainMnist ${CMAKE_CURRENT_SOURCE_DIR}/examples/train_mnist.cpp)
target_include_directories(TrainMnist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mojo)
target_compile_options(TrainMnist PUBLIC -std=c++11 -O3)


add_executable(TestMnistAVX512 ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
target_include_directories(TestMnistAVX512 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mojo)
target_link_libraries(TestMnistAVX512 PUBLIC ${OpenMP_CXX_LIBRARIES})
target_compile_definitions(TestMnistAVX512 PUBLIC -DMOJO_AVX512 -DMOJO_OMP)
target_compile_options(TestMnistAVX512 PUBLIC -std=c++11 -fopenmp -O3 -mavx512f)

add_executable(TestMnist ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
target_include_directories(TestMnist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mojo)
target_compile_options(TestMnist PUBLIC -std=c++11 -O3)
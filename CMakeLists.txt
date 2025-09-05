cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler before project()
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(chimera_scalper LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture for RTX 4070 Ti
set(CMAKE_CUDA_ARCHITECTURES "89")

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(SOURCES
        src/main_bybit.cpp
    src/core/data_feed.cpp
    src/core/bybit_feed.cpp
    src/core/features.cu
    src/layers/layer1_fastpath.cpp
    src/layers/layer2_regime.cpp
    src/layers/layer3_experts.cpp
    src/layers/layer4_pnlfilter.cpp
    src/layers/layer5_execution.cpp
    src/utils/logger.cpp
)

# Create executable
add_executable(chimera_scalper ${SOURCES})

# Link libraries
target_link_libraries(chimera_scalper
    CUDA::cudart
    pthread
)

# Set output directory
set_target_properties(chimera_scalper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

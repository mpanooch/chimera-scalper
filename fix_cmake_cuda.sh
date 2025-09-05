#!/bin/bash

echo "🔧 Fixing CMake CUDA Detection"
echo "=============================="

# Check CUDA installation
echo "Checking CUDA installation..."
which nvcc
nvcc --version

echo ""
echo "Setting up environment variables..."

# Set CUDA paths explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDACXX=$CUDACXX"

# Create a new CMakeLists.txt that explicitly sets CUDA paths
cat > CMakeLists.txt << 'EOF'
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
    src/main.cpp
    src/core/data_feed.cpp
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
EOF

echo "✓ Created fixed CMakeLists.txt"
echo ""

# Try building again
echo "🔨 Attempting build with explicit CUDA paths..."
rm -rf build
mkdir build
cd build

# Run cmake with explicit CUDA compiler
cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    2>&1 | tee cmake_output.log

# Check if CMake succeeded
if [ -f "Makefile" ]; then
    echo ""
    echo "✅ CMake configuration successful!"
    echo "Building project..."

    make -j$(nproc) 2>&1 | tee make_output.log

    if [ -f "bin/chimera_scalper" ]; then
        echo ""
        echo "🎉 BUILD SUCCESSFUL!"
        echo "Run with: ./bin/chimera_scalper"
    else
        echo ""
        echo "❌ Make failed. Checking errors..."
        echo "Last 20 lines of make output:"
        tail -20 make_output.log
    fi
else
    echo ""
    echo "❌ CMake configuration failed"
    echo "Trying alternative approach..."

    # Alternative: Build without CMake
    cd ..
    echo ""
    echo "Building without CMake (manual compilation)..."

    mkdir -p manual_build
    cd manual_build

    # Compile CUDA file
    echo "Compiling features.cu..."
    nvcc -c ../src/core/features.cu -o features.o \
        -I../src -I/usr/local/cuda/include \
        --std=c++17 -arch=sm_89

    # Compile C++ files
    echo "Compiling C++ files..."
    g++ -c ../src/main.cpp -o main.o \
        -I../src -I/usr/local/cuda/include \
        -std=c++17

    g++ -c ../src/core/data_feed.cpp -o data_feed.o \
        -I../src -std=c++17

    g++ -c ../src/layers/layer1_fastpath.cpp -o layer1_fastpath.o \
        -I../src -std=c++17

    g++ -c ../src/layers/layer2_regime.cpp -o layer2_regime.o \
        -I../src -std=c++17

    g++ -c ../src/layers/layer3_experts.cpp -o layer3_experts.o \
        -I../src -std=c++17

    g++ -c ../src/layers/layer4_pnlfilter.cpp -o layer4_pnlfilter.o \
        -I../src -std=c++17

    g++ -c ../src/layers/layer5_execution.cpp -o layer5_execution.o \
        -I../src -std=c++17

    g++ -c ../src/utils/logger.cpp -o logger.o \
        -I../src -std=c++17

    # Link everything
    echo "Linking..."
    g++ -o chimera_scalper \
        main.o features.o data_feed.o \
        layer1_fastpath.o layer2_regime.o layer3_experts.o \
        layer4_pnlfilter.o layer5_execution.o logger.o \
        -L/usr/local/cuda/lib64 -lcudart -lpthread

    if [ -f "chimera_scalper" ]; then
        echo ""
        echo "✅ Manual build successful!"
        echo "Running CHIMERA..."
        ./chimera_scalper
    else
        echo "❌ Manual build also failed"
        echo "Checking for specific errors..."

        # Check which file is causing issues
        for obj in *.o; do
            if [ ! -f "$obj" ]; then
                echo "Missing: $obj"
            fi
        done
    fi
fi
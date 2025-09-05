#!/bin/bash

# CHIMERA Scalper Build Script
# Optimized for RTX 4070 Ti (sm_89) with CUDA 12.x

set -e  # Exit on error

echo "🔥 CHIMERA Build System"
echo "======================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}❌ CUDA not found! Please install CUDA toolkit.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CUDA found:${NC} $(nvcc --version | head -n 1)"

# Create build directory
mkdir -p build
cd build

# Generate L2 data if needed
if [ ! -d "../data/ob" ] || [ -z "$(ls -A ../data/ob 2>/dev/null)" ]; then
    echo -e "${YELLOW}📊 Generating synthetic L2 order book data...${NC}"
    cd ..
    python generate_synthetic_l2.py
    cd build
else
    echo -e "${GREEN}✓ L2 data already exists${NC}"
fi

# Configure with CMake
echo -e "${YELLOW}🔧 Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Build
echo -e "${YELLOW}🔨 Building CHIMERA...${NC}"
make -j$(nproc)

# Run tests if build successful
if [ -f "bin/chimera_scalper" ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    echo -e "${YELLOW}🧪 Running microstructure engine test...${NC}"
    ./bin/test_microstructure

    echo ""
    echo -e "${GREEN}🚀 CHIMERA is ready!${NC}"
    echo -e "   Run: ${YELLOW}./bin/chimera_scalper${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
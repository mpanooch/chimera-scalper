#!/bin/bash

# CHIMERA Scalper Setup Script
# This script creates the complete directory structure

set -e

echo "🔥 CHIMERA Scalper Setup"
echo "========================"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."

mkdir -p src/core
mkdir -p src/layers
mkdir -p src/utils
mkdir -p data/ob
mkdir -p models
mkdir -p mathematica
mkdir -p build
mkdir -p dev

echo "✓ Directories created"

# Now you need to manually create/copy the files to their locations
echo ""
echo "📝 Please create the following files in their respective locations:"
echo ""
echo "ROOT directory files:"
echo "  • CMakeLists.txt"
echo "  • build.sh (make it executable: chmod +x build.sh)"
echo "  • generate_synthetic_l2.py"
echo ""
echo "src/ directory:"
echo "  • main.cpp"
echo ""
echo "src/core/ directory:"
echo "  • features.h"
echo "  • features.cu"
echo "  • data_feed.h"
echo "  • data_feed.cpp"
echo ""
echo "src/layers/ directory:"
echo "  • layer1_fastpath.h"
echo "  • layer1_fastpath.cpp"
echo "  • layer2_regime.h"
echo "  • layer2_regime.cpp"
echo "  • layer3_experts.h"
echo "  • layer3_experts.cpp"
echo "  • layer4_pnlfilter.h"
echo "  • layer4_pnlfilter.cpp"
echo "  • layer5_execution.h"
echo "  • layer5_execution.cpp"
echo ""
echo "src/utils/ directory:"
echo "  • logger.h"
echo "  • logger.cpp"
echo "  • config.h"
echo "  • timer.h"
echo ""
echo "After creating all files, run:"
echo "  1. python generate_synthetic_l2.py  # Generate L2 data"
echo "  2. ./build.sh                        # Build the system"
echo ""

# Check for required dependencies
echo "🔍 Checking dependencies..."

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found: $(nvcc --version | head -n 1)"
else
    echo "❌ CUDA not found! Please install CUDA toolkit."
fi

if command -v cmake &> /dev/null; then
    echo "✓ CMake found: $(cmake --version | head -n 1)"
else
    echo "❌ CMake not found! Please install CMake."
fi

if command -v python &> /dev/null; then
    echo "✓ Python found: $(python --version)"
else
    echo "❌ Python not found!"
fi

# Check for Eigen3
if pkg-config --exists eigen3; then
    echo "✓ Eigen3 found"
else
    echo "⚠️  Eigen3 not found. Install with: sudo pacman -S eigen"
fi

echo ""
echo "✅ Setup complete! Follow the file creation instructions above."
#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "CHIMERA Test\n";

    int devices = 0;
    cudaGetDeviceCount(&devices);
    std::cout << "CUDA Devices: " << devices << "\n";

    if (devices > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << "\n";
    }

    return 0;
}

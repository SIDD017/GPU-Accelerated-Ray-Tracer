#include "tracer.cuh"
#include "defines.cuh"

namespace CUDA_Tracer {

Tracer::Tracer(int nx, int ny) {
    this->nx = nx;
    this->ny = ny;
    this->num_pixels = nx * ny;
    fb_size = 3 * num_pixels * sizeof(float);
    CHECK_CUDA_ERRORS(cudaMallocManaged((void **)&fb, fb_size));
}

Tracer::~Tracer() {}

void Tracer::draw(int tx, int ty) {
    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << " in " << tx << "x" << ty << " blocks.\n";

    clock_t start, stop;
    start = clock();

    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(nx, ny, fb);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took" << timer_seconds << " seconds.\n";
    output_image();
}

__global__
void render(int nx, int ny, float *fb) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) {
         return;
    }
    int pixel_index = j * nx * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / nx;
    fb[pixel_index + 1] = float(j) / ny;
    fb[pixel_index + 2] = 0.2f;
}

void Tracer::output_image() {
    std::cout  << "P3\n" << nx << " " << ny << "\n255\n";
    for(int j = ny - 1; j >= 0;j--) {
        for(int i = 0; i < nx; i++) {
            size_t pixel_index = j * 3 * nx + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}
}
#include <stdio.h>
#include "defines.cuh"

namespace CUDA_Tracer {
class Tracer {
    private:
    int nx;
    int ny;
    int num_pixels;
    size_t fb_size;
    float *fb;

    public:
    Tracer(int nx, int ny);
    ~Tracer();
    void draw(int tx, int ty);
    __global__ void render();
    void output_image();
};
}